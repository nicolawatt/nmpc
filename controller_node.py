#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import csv
import socket
import json
import os
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from transforms3d.euler import quat2euler

from .controller_class import Controller  

class NMPCNode(Node):
    def __init__(self):
        super().__init__('nmpc_controller_node')
        
        # Load parameters
        self.declare_parameter('rate', 10)
        self.declare_parameter('trajectory_file', '/home/voyager/data/nicola_watt/trajectories/recorded_odometry.csv')
        self.declare_parameter('min_v', -1.0)
        self.declare_parameter('max_v', 1.0)
        self.declare_parameter('min_w', -np.pi/2)
        self.declare_parameter('max_w', np.pi/2)
        
        self.rate = self.get_parameter('rate').value
        self.csv_file = self.get_parameter('trajectory_file').value
        
        # Initialize data storage
        self.actual_trajectory = []
        self.reference_trajectory_log = []
        
        # Load trajectory from CSV
        self.reference_trajectory = self.load_trajectory()
        
        # Controller parameters
        init_pos = [0, 0, 0]  # initial x, y, theta
        min_v = self.get_parameter('min_v').value
        max_v = self.get_parameter('max_v').value
        min_w = self.get_parameter('min_w').value
        max_w = self.get_parameter('max_w').value
        
        # Initialize controller
        self.controller = Controller(init_pos, min_v, max_v, min_w, max_w, T=1.0/self.rate, N=20)
        
        # ROS publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, 'odometry', self.odom_callback, 10)
        
        self.current_state = np.array(init_pos)
        self.trajectory_index = 0

        # Timer for control loop
        self.timer = self.create_timer(1.0/self.rate, self.control_loop)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.plotter_address = ('196.24.152.120', 12345)  


    def load_trajectory(self):
        trajectory = []
        with open(self.csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                x, y, theta = map(float, row)
                trajectory.append([x, y, theta])
        return np.array(trajectory)

    def odom_callback(self, msg):
        # Extract position
        position = msg.pose.pose.position
        
        # Extract orientation and convert to euler angles
        orientation = msg.pose.pose.orientation
        _, _, yaw = quat2euler([orientation.x, orientation.y, orientation.z, orientation.w])
        
        self.current_state = np.array([position.x, position.y, yaw])

    def get_reference_trajectory(self):
        N = self.controller.N
        remaining_points = len(self.reference_trajectory) - self.trajectory_index
        
        if remaining_points >= N + 1:
            ref_traj = self.reference_trajectory[self.trajectory_index:self.trajectory_index+N+1]
        else:
            ref_traj = np.vstack((
                self.reference_trajectory[self.trajectory_index:],
                np.tile(self.reference_trajectory[-1], (N+1-remaining_points, 1))
            ))
        
        self.trajectory_index += 1
        if self.trajectory_index >= len(self.reference_trajectory):
            self.trajectory_index = 0  # Loop back to start
        
        return ref_traj

    def send_data(self):
        trajectory_data ={
            'actual_x' : float(self.current_state[0]),
            'actual_y' : float(self.current_state[1]),
            'forecast_x': self.controller.next_states[:, 0].tolist() if hasattr(self.controller, 'next_states') else [],  # Convert to list
            'forecast_y': self.controller.next_states[:, 1].tolist() if hasattr(self.controller, 'next_states') else [],  # Convert to list
        }

        self.socket.sendto(json.dumps(trajectory_data).encode(), self.plotter_address)

    def control_loop(self):
        # Get reference trajectory for the next N steps
        ref_trajectory = self.get_reference_trajectory()

        # Compute reference controls (you may need to adjust this based on your needs)
        ref_controls = np.zeros((self.controller.N, 2))
        
        # Solve NMPC
        optimal_control = self.controller.solve(ref_trajectory, ref_controls)
        
        print("Optimal v:")
        print(optimal_control[0])
        print("Optimal w:")
        print(optimal_control[1])

        # Create and publish Twist message
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = float(optimal_control[0])  # v
        cmd_vel_msg.angular.z = float(optimal_control[1])  # w
        self.cmd_vel_pub.publish(cmd_vel_msg)

        self.send_data()
        
def destroy_node(self):
        self.socket.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    nmpc_node = NMPCNode()
    rclpy.spin(nmpc_node)
    nmpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
