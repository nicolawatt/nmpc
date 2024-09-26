import casadi as ca 
import numpy as np
import math

class Controller:
    def __init__(self, init_pos, min_v, max_v, min_w, max_w, T=0.02, N=20):
        self.T = T
        self.N = N

        self.Q = np.diag([50, 50, 10])
        self.R = np.diag([1, 1])

        #costraints
        self.min_v = min_v
        self.max_v = max_v

        self.min_w = min_w
        self.max_w = max_w

        self.max_dv = 0.8
        self.max_dw = math.pi/6

        #history states and controls
        self.next_states = np.ones((self.N+1, 3))*init_pos
        self.u0 = np.zeros((self.N, 2))

        self.setup_controller()

    def setup_controller(self):
        self.opti = ca.Opti()

        #state: x, y, theta
        self.opt_states = self.opti.variable(self.N+1, 3)
        x = self.opt_states[:,0]
        y = self.opt_states[:,1]
        theta = self.opt_states[:,2]

        #control inputs: v, w
        self.opt_controls = self.opti.variable(self.N, 2)
        v = self.opt_controls[0]
        w = self.opt_controls[1]

        #model
        f = lambda x_, u_: ca.vertcat(*[
            ca.cos(x_[2])*u_[0],    #dx = v*cos(theta)
            ca.sin(x_[2])*u_[0],    #dy = vsin(theta)
            u_[1]                   #dtheta = w
        ])

        #parameters 
        self.opt_u_ref = self.opti.parameter(self.N, 2)
        self.opt_x_ref = self.opti.parameter(self.N+1, 3)

        #initial condition
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_ref[0, :])
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T*self.T
            self.opti.subject_to(self.opt_states[i+1, :] == x_next)

        #cost function
        obj = 0
        for i in range(self.N):
            stateerror = self.opt_states[i, :] - self.opt_x_ref[i+1, :]
            controlerror = self.opt_controls[i, :] - self.opt_u_ref[i, :]
            obj = obj + ca.mtimes([stateerror, self.Q, stateerror.T]) \
                        + ca.mtimes([controlerror, self.R, controlerror.T])
        self.opti.minimize(obj)

        #constraints
        for i in range(self.N-1):
            dvel = (self.opt_controls[i+1, :] - self.opt_controls[i, :])/self.T
            self.opti.subject_to(self.opti.bounded(-self.max_dv, dvel[0], self.max_dv))
            self.opti.subject_to(self.opti.bounded(-self.max_dw, dvel[1], self.max_dw))

        #boundary and control conditions
        self.opti.subject_to(self.opti.bounded(self.min_v, v, self.max_v))
        self.opti.subject_to(self.opti.bounded(self.min_w, w, self.max_w))

        #configure solver (IPOPT)
        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-8,
                        'ipopt.acceptable_obj_change_tol':1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, next_trajectories, next_controls):
        #updates reference values
        self.opti.set_value(self.opt_x_ref, next_trajectories)
        self.opti.set_value(self.opt_u_ref, next_controls)

        #initial guess for optimisation
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0)

        #solve optimisation
        sol = self.opti.solve()

        #return optimal control input
        self.u0 = sol.value(self.opt_controls)
        self.next_states = sol.value(self.opt_states)

        return self.u0[0, :]
