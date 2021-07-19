import numpy as np
from scipy.spatial.transform import Rotation as R
import operator
from Config import Config


class Agent():
    def __init__(self, start_x, start_y, start_z, goal_x, goal_y, goal_z, id):

        # Global Frame states
        self.pos_global_frame = np.array([start_x, start_y, start_z], dtype='float64')
        self.goal_global_frame = np.array([goal_x, goal_y, goal_z], dtype='float64')
        self.vel_global_frame = np.array([0, 0, 0], dtype='float64')

        # Other parameters
        self.radius = Config.radius
        self.MaxVelo = Config.MaxVelo
        self.MaxAcc = Config.MaxAcc
        self.id = id
        self.dist_to_goal = 0.0

        self.pitch = 0.0
        self.roll = 0.0
        self.pitch_torq = 0.0
        self.roll_torq = 0.0
        self.force_dw = np.array([0.0, 0.0, 0.0])
        self.thrust_m = 9.81 # thrust per mass
        self.R_BI = Config.world # rotation matrix from body to world

    def update_info(self, px, py, pz, vx, vy, vz, gx, gy, gz):
        self.pos_global_frame = np.array([px, py, pz], dtype='float64')
        self.vel_global_frame = np.array([vx, vy, vz], dtype='float64')       
        self.goal_global_frame = np.array([gx, gy, gz], dtype='float64')

    def update_dw(self, force, roll_ddot, pitch_ddot):
        self.force_dw = force
        self.roll_torq = roll_ddot
        self.pitch_torq = pitch_ddot
        # print(force)

    def update_state(self, force, dt, MaxAcc, MaxVelo):
        self.MaxVelo = MaxVelo
        self.MaxAcc = MaxAcc
        
        # Limit vertical acceleration
        force[2] = max(min(force[2], self.MaxAcc), -self.MaxAcc)
        # if force[2] > 10.0:     force[2] = 10.0
        # elif force[2] < -9.80:  force[2] = -9.80

        if Config.DYN:
            max_ang = 0.3
            # Command/desired angles
            # pitch_c = np.arctan2(-force[0], force[2]+9.81)
            # roll_c = np.arctan2(force[1], force[2]+9.81)
            pitch_c = np.arctan2(-force[1], force[2]+9.81)  # world frame to body frame
            roll_c = np.arctan2(force[0], force[2]+9.81)

            # To limit the velocity
            if self.vel_global_frame[2] >= self.MaxVelo:     force[2] = -9.80
            elif self.vel_global_frame[2] <= -self.MaxVelo:  force[2] = 10.0
            
            if self.vel_global_frame[0] >= self.MaxVelo:     roll_c = -max_ang
            elif self.vel_global_frame[0] <= -self.MaxVelo:  roll_c = max_ang
            else: roll_c = np.arctan2(force[0], force[2]+9.81)

            if self.vel_global_frame[1] >= self.MaxVelo:     pitch_c = max_ang
            elif self.vel_global_frame[1] <= -self.MaxVelo:  pitch_c = -max_ang
            else: pitch_c = np.arctan2(-force[1], force[2]+9.81)

            if pitch_c == np.pi or pitch_c == -np.pi:   pitch_c = 0.0
            if roll_c == np.pi or roll_c == np.pi:      roll_c = 0.0

            pitch_d = -5.5759*self.pitch + 5.5759*pitch_c
            roll_d = -5.5759*self.roll + 5.5759*roll_c

            self.pitch += pitch_d*dt
            self.roll += roll_d*dt
            
            # Limit roll and pitch angles
            self.pitch = max(min(self.pitch, max_ang), -max_ang)
            self.roll = max(min(self.roll, max_ang), -max_ang)
            # if (self.pitch > max_ang):    self.pitch = max_ang
            # elif (self.pitch < -max_ang): self.pitch = -max_ang
            # if (self.roll > max_ang):     self.roll = max_ang
            # elif (self.roll < -max_ang):  self.roll = -max_ang
                
            # Consider the torque/roll/pitch produced by downwash
            if Config.TORQUE_FLIP:
                if self.roll_torq != 0.0 or self.pitch_torq != 0.0:
                    self.pitch += 0.5*self.pitch_torq*dt*dt
                    self.roll += 0.5*self.roll_torq*dt*dt
                    self.roll_torq, self.pitch_torq = 0.0, 0.0

            force[1] = -(9.81+force[2])*np.tan(self.pitch)
            force[0] = (9.81+force[2])*np.cos(self.pitch)*np.tan(self.roll) - force[1]*np.sin(self.pitch)*np.tan(self.roll)
        
            self.thrust_m = -np.sin(self.pitch)*force[0] + np.cos(self.pitch)*np.sin(self.roll)*force[1] + np.cos(self.pitch)*np.cos(self.roll)*(force[2] + 9.81)
            M1 = np.array([[np.cos(self.roll), 0, np.sin(self.roll)],[0, 1, 0],[-np.sin(self.roll), 0, np.cos(self.roll)]])
            M2 = np.array([[1, 0, 0],[0, np.cos(self.pitch), -np.sin(self.pitch)],[0, np.sin(self.pitch), np.cos(self.pitch)]])
            M = np.dot(M2, M1)
            self.R_BI = np.dot(M, Config.world)
        else:
            force[0] = max(min(force[0], self.MaxAcc), -self.MaxAcc)
            force[1] = max(min(force[1], self.MaxAcc), -self.MaxAcc)
            # if force[0] > max_acc:     force[0] = max_acc
            # elif force[0] < -max_acc:  force[0] = -max_acc
            # if force[1] > max_acc:     force[1] = max_acc
            # elif force[1] < -max_acc:  force[1] = -max_acc
        # print('out', force)

        self.vel_global_frame += force * dt
        if not Config.DYN:
            self.vel_global_frame[0] = max(min(self.vel_global_frame[0], self.MaxVelo), -self.MaxVelo)
            self.vel_global_frame[1] = max(min(self.vel_global_frame[1], self.MaxVelo), -self.MaxVelo)
            self.vel_global_frame[2] = max(min(self.vel_global_frame[2], self.MaxVelo), -self.MaxVelo)
            # if (np.linalg.norm(self.vel_global_frame) > self.MaxVelo): # CapVelocity
            #     self.vel_global_frame = (self.vel_global_frame / np.linalg.norm(self.vel_global_frame)) * self.MaxVelo

        self.pos_global_frame += self.vel_global_frame*dt

        # Consider the downwash effect
        if (self.force_dw != np.array([0.0, 0.0, 0.0])).any():
            self.vel_global_frame += self.force_dw * dt
            self.pos_global_frame += self.vel_global_frame*dt
            print(self.id, force_dw)
            self.force_dw = np.array([0.0, 0.0, 0.0])

        self.dist_to_goal = np.linalg.norm(self.goal_global_frame - self.pos_global_frame)




