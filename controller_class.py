import casadi as ca 
import numpy as np
import math

class Controller:
    def __init__(self, min_v, max_v, min_w, max_w, T, N):
    #initialise nmpc controller
        #time step
        self.T = T

        #horizon length
        self.N = N

        #cost function weight matrices
        self.Q = np.diag([100, 100, 10])
        self.R = np.diag([1, 1])

        #linear and angular velocity costraints
        self.min_v = min_v
        self.max_v = max_v
        self.min_w = min_w
        self.max_w = max_w

        #linear and angular acceleration constraints
        self.max_dv = 0.8
        self.max_dw = math.pi/6

        #history of states and controls
        self.next_states = None
        self.u0 = np.zeros((self.N, 2))

        #setup optimisation problem
        self.setup_controller()

    def setup_controller(self):
    #function to setup optimisation problem using CasADi Opti
        self.opti = ca.Opti()

        #state variables: x, y, theta
        self.opt_states = self.opti.variable(self.N+1, 3)
        x = self.opt_states[:, 0]
        y = self.opt_states[:, 1]
        theta = self.opt_states[:, 2]

        #control variables: v, w
        self.opt_controls = self.opti.variable(self.N, 2)
        v = self.opt_controls[:, 0]
        w = self.opt_controls[:, 1]

        #kinematic model for differential drive robot
        f = lambda x_, u_: ca.vertcat(*[
            ca.cos(x_[2])*u_[0],    #dx = v*cos(theta)
            ca.sin(x_[2])*u_[0],    #dy = vsin(theta)
            u_[1]                   #dtheta = w
        ])

        #parameters (reference trajectory and initial state)
        self.opt_u_ref = self.opti.parameter(self.N, 2)
        self.opt_x_ref = self.opti.parameter(self.N+1, 3)
        self.opt_x0 = self.opti.parameter(3, 1)

        #set initial condition
        self.opti.subject_to(self.opt_states[0, :].T == self.opt_x0)
        
        #system dynamics over prediction horizon
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T*self.T
            self.opti.subject_to(self.opt_states[i+1, :] == x_next)

        #cost function
        obj = 0
        for i in range(self.N):
            state_error = self.opt_states[i, :] - self.opt_x_ref[i+1, :]
            control_error = self.opt_controls[i, :] - self.opt_u_ref[i, :]
            obj = obj + ca.mtimes([state_error, self.Q, state_error.T]) #+ ca.mtimes([control_error, self.R, control_error.T])
        self.opti.minimize(obj)

        #change in control input constraints
        for i in range(self.N-1):
            dvel = (self.opt_controls[i+1, :] - self.opt_controls[i, :])/self.T
            self.opti.subject_to(self.opti.bounded(-self.max_dv, dvel[0], self.max_dv))
            self.opti.subject_to(self.opti.bounded(-self.max_dw, dvel[1], self.max_dw))

        #control input constraints
        self.opti.subject_to(self.opti.bounded(self.min_v, v, self.max_v))
        self.opti.subject_to(self.opti.bounded(self.min_w, w, self.max_w))

        #configure solver (IPOPT)
        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-8,
                        'ipopt.acceptable_obj_change_tol':1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, current_state, next_trajectories, next_controls):
    #function to solve nmpc optimisation problem
        #update parameters with current state and reference values
        self.opti.set_value(self.opt_x0, np.array(current_state).reshape(3, 1))
        self.opti.set_value(self.opt_x_ref, next_trajectories)
        self.opti.set_value(self.opt_u_ref, next_controls)

        #initialise or update predicted states
        if self.next_states is None or np.allclose(self.next_states[0], current_state) == False:
            self.next_states = np.zeros((self.N+1, 3))
            self.next_states[0] = current_state
            for i in range(1, self.N+1):
                self.next_states[i] = next_trajectories[i]

        #initial guess for optimisation
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0)

        #solve optimisation
        sol = self.opti.solve()

        #retrieve optimal control input
        self.u0 = sol.value(self.opt_controls)
        self.next_states = sol.value(self.opt_states)

        #return first optimal control input
        return self.u0[0, :]
