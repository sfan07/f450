import numpy as np
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation
from numpy.linalg import norm
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize
from scipy.interpolate import UnivariateSpline
from collision_avoidance import CollisionAvoidance
import math
from pso_updatingLocation import path_generation
from numpy.core.defchararray import title
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.cm as cm
from vpython import cylinder, vector
from scipy.interpolate import interp1d
import numpy.matlib
import random
import copy
import time
from Config import Config
# from xlsxwriter import Workbook

class PID():
    def __init__(self, kp, ki, kd, u_min, u_max, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.u_min = u_min
        self.u_max = u_max
        self.dt = dt
        self.e_prev = None
        self.e_accum = 0
        
    def control(self,ref,state):
        e = ref - state
        self.e_accum += e
        #self.e_accum = np.clip(self.e_accum, self.i_min, self.i_max)
        if self.e_prev is None:
            self.e_prev = e

        u = self.kp*e + self.ki*self.e_accum + self.kd*(e - self.e_prev)/self.dt
        u = np.clip(u, self.u_min, self.u_max)

        self.e_prev = e
        return u
        
    def reset(self):
        self.e_prev = None
        self.e_accum = 0

class Sim2D():   
    def __init__(self, agent_pos, agent_vel, agent_acc, agent_ang, target_pos, obs_init, dt):
        '''
        Parameters
        ----------
        agent_init: (4+)xN array, each column is [x,y,vx,vy,etc...].T
        target_init: 4x1 array [x,y,vx,vy].T
        obs_init: 5xN array, each column is [x,y,z,vx,vy,r].T, or None [x,y,z,h,r,vx,vy].T
        dt: timestep for discrete model
        order: approximation order for crazyflie dynamics, default is 2
        '''
        # full state is [x,y,vx,vy,roll,pitch,roll_dot,pitch_dot]
        # assume second order approximation
        # A = np.zeros([8,8]) 
        # A[0,2], A[1,3] = 1, 1
        # A[2,5], A[3,4] = 9.81*np.pi/180, -9.81*np.pi/180 
        # A[4,6], A[5,7] = 1, 1
        # A[6,4], A[7,5] = -37.2808, -37.2808
        # A[6,6], A[7,7] = -8.1488, -8.1488

        # B = np.vstack((np.zeros([6,2]), np.diag([37.2808, 37.2808])))
        
        # # handle zero and first order cases
        # if order == 0:
        #     A = A[0:4,0:4]
        #     B = np.vstack((np.zeros([2,2]), np.eye(2)))
    
        # if order == 1:
        #     A = A[0:6,0:6]
        #     A[4,4], A[5,5] = -5.5759, -5.5759
        #     B = np.vstack((np.zeros([4,2]), np.diag([5.5759, 5.5759])))

        # discretized dynamics
        # ncols = A.shape[1] + B.shape[1]
        # H = np.zeros([ncols,ncols])
        # H[0:A.shape[0],0:A.shape[0]] = A*dt
        # H[0:B.shape[0],A.shape[0]:] = B*dt
        # H = expm(H)
        
        # self.order = order
        self.num_agents = agent_pos.shape[0]
        self.num_obstacles = 0 if obs_init is None else obs_init.shape[1]
        # self.num_iter = num_iter
        self.dt = dt
        # self.A = A
        # self.B = B
        
        # self.A_d = H[0:A.shape[0],0:A.shape[0]]
        # self.B_d = H[0:B.shape[0],A.shape[0]:]
        # if self.order == 2:
        #     self.K = np.array([[       0,      10,        0, 21.6188, -2.1450,       0, -0.8598,       0],
        #                        [     -10,       0, -21.6188,       0,       0, -2.1450,       0, -0.8598]])
        # elif self.order == 1:
        #     self.K = np.array([[       0,      10,        0, 17.4568, -0.7527,       0],
        #                        [     -10,       0, -17.4568,       0,       0, -0.7527]])
    
        # self.agent_init = np.copy(agent_init)
        # self.agent_state = np.copy(agent_init)
        # self.target_init = np.copy(target_init)
        # self.target_state = np.copy(target_init)
        self.target_pos = np.copy(target_pos)
        self.agent_pos = np.copy(agent_pos)
        self.agent_pos_next = np.copy(agent_pos)
        self.agent_acc = np.copy(agent_acc)
        self.agent_ang = np.copy(agent_ang)
        self.agent_vel = np.copy(agent_vel)
        self.obs_init = None if obs_init is None else np.copy(obs_init)
        self.obs_state = None if obs_init is None else np.copy(obs_init)

        # obstacle state
        self.xobs = None if obs_init is None else self.obs_init[0,:self.num_obstacles] 
        self.yobs = None if obs_init is None else self.obs_init[1,:self.num_obstacles]
        self.zobs = None if obs_init is None else self.obs_init[2,:self.num_obstacles]
        self.robs = None if obs_init is None else self.obs_init[3,:self.num_obstacles]
        self.hobs = None if obs_init is None else self.obs_init[4,:self.num_obstacles]
        self.xvobs = None if obs_init is None else self.obs_init[5,:self.num_obstacles]
        self.yvobs = None if obs_init is None else self.obs_init[6,:self.num_obstacles]
        self.nObs = 0 if obs_init is None else len(self.xobs)
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = -3,3, -3, 3, 0, 2
        c1_alp = 3 #3
        c2_alp = 1
        d = 0.4
        d_p = 0.3
        r_bet = 0.5
        h_bet = 0.5
        eps = 0.1
        r = 0.3
        A_UAV = np.pi*((1/2)*0.092)**2
        h_alpha = 2
        dw_h = 2
        self.avoidance_controller = CollisionAvoidance(c1_alp, c2_alp, d, d_p, r_bet, h_bet, eps, r, A_UAV, h_alpha, dw_h)
        
    def ifCollision(self):
        brkflg = 0
        for n in range(len(self.obs_state[0])):
            for i in range(len(self.agent_state[0])):
                if (np.linalg.norm(self.obs_state[:2,n]-self.agent_state[:2,i])<self.obs_state[3,n]):
                    print(f'COLLISION between agent{i} and obstacle{n} OCCURs!')
                    print(f'Current Agent{i} Position', self.agent_state[:,:3])
                    brkflg = 1
                    break
            if brkflg == 1:
                break
            
    def step(self, agent_input, target_input, obs_input, return_copy = True):
        '''
        Parameters
        ----------
        agent_input : 2xN inputs, each column is desired [roll, pitch].T
                      or accelerations [ax, ay].T if using 0 order assumption
        target_input : 2x1 velocity, [vx, vy].T
        obs_input : 2xN velocities, each column is [vx, vy].T
        
        If any inputs are None, update occurs assuming 0 input
​        
        Returns
        -------
        References or copies of [agent_state, target_state, obs_state]
        '''
        self.agent_state = self.A_d @ self.agent_state
        if agent_input is not None:
            self.agent_state += self.B_d @ agent_input
            
        if target_input is not None:
            # self.target_state[:,2:] = np.copy(target_input)
            self.target_state[2:] = np.copy(target_input)
        self.target_state[0:2] += self.target_state[2:]*self.dt
        
        if self.obs_state is not None:
            if obs_input is not None:
                self.obs_state[5:] = np.copy(obs_input)
            self.obs_state[0:2] += self.obs_state[5:]*self.dt

        self.ifCollision()    

        if return_copy:
            return [np.copy(self.agent_state), 
                    np.copy(self.target_state),
                    None if self.obs_state is None else np.copy(self.obs_state)]
        else:
            return [self.agent_state, self.target_state, self.obs_state]

    def get_vertices(self, l):
        xt, yt, zt = self.target_pos[0][0],self.target_pos[0][1],self.target_pos[0][2]
        vertices = np.array([[xt + l, yt + l, zt + l],
                             [xt + l, yt - l, zt - l],
                             [xt - l, yt + l, zt - l],
                             [xt - l, yt - l, zt + l]])
        return vertices

    def hungarian_assignment(self, vertex_vec, agent_vec):
        cost_mtx = -1*(vertex_vec@agent_vec.T)
        assignment = linear_sum_assignment(cost_mtx)[1]
        return assignment

    def get_cross_mtx(self, vec):
        x, y, z = vec
        cross_mtx = np.array([[0, -z, y],
                              [z, 0, -x],
                              [-y, x, 0]])
        return cross_mtx

    def rotation_from_axis_angle(self, axis, angle):
        n_cross = self.get_cross_mtx(axis)
        C = np.eye(3)+np.sin(angle)*n_cross+(1-np.cos(angle))*n_cross@n_cross
        return C
                
    def run(self, l, look_ahead_num, look_ahead_dt):
        '''
        h: height of the target
        l: sizing coefficient of the vertices
        look_ahead_num: points for spline interpolation
        look_ahead_dt: dt for spline interpolation
        '''
        # augment agent and target state with height
        # self.agent_pos = np.hstack((self.agent_state.T[:,:2], np.zeros((self.num_agents,1)))) # [x,y,0]
        # self.target_pos = np.hstack((self.target_init.T[:2], h)) #[x,y,height of target]

        # get vertex position and vectors
        # print(self.target_pos)
        self.vertex_pos = self.get_vertices(l)
        self.vertex_vec = normalize(self.vertex_pos-self.target_pos, axis = 1, norm = 'l2')

        # height of the agents, starting from 0, each row is [z, vz]
        #self.agent_h = np.zeros((self.num_agents, 2))
        self.agent_h = np.column_stack((self.agent_pos[:,2], self.agent_vel[:,2]))

        # spline interpolation parameters 
        self.look_ahead_num = look_ahead_num 
        self.look_ahead_dt = look_ahead_dt

        #################################################################
        # for plotting/verification
        # self.agent_coords = np.zeros((self.num_agents, self.num_iter, 3))
        #################################################################

        # self.reached = np.array([False]*self.num_agents)
        Total_time = 0
        pso_funccall_time = 0
        agent_input = np.zeros((2, self.num_agents)) #desired [roll, pitch].T

        #for spline interpolation
        ts = np.zeros((self.look_ahead_num, self.num_agents))
        xs = np.zeros((self.look_ahead_num, self.num_agents))
        ys = np.zeros((self.look_ahead_num, self.num_agents))
        zs = np.zeros((self.look_ahead_num, self.num_agents))
        look_ahead_pts = np.zeros((self.look_ahead_num+1, self.num_agents, 3))
        look_ahead_pts[0] = self.agent_pos[:,:3]   #agent_pos [x,y,z]

        collision_input = np.zeros((self.num_agents, 6)) #[x,y,h,vx,vy,vh]
        collision_input = np.column_stack((self.agent_pos[:,:3],self.agent_vel[:,:3]))
        collision_acc, downwash_flag, neighbors_pos = self.avoidance_controller.get_control_all(collision_input) # collision_input: [x,y,z,vx,vy,vz]
        agent_coord = collision_input[:,:3]#[x,y,h]
        
        # compute the next look_ahead points for interpolation
        self.agent_vec = normalize(look_ahead_pts[0]-self.target_pos, axis = 1, norm = 'l2')
        # Hungarian Algorithm for vertex assignment
        assignment = self.hungarian_assignment(self.vertex_vec, self.agent_vec)
        self.assigned_vertex_pos = self.vertex_pos[assignment]
        self.assigned_vertex_vec = self.vertex_vec[assignment]

        self.agent_posx = self.agent_pos[:,0]
        self.agent_posy = self.agent_pos[:,1]
        self.agent_posz = self.agent_pos[:,2]

        self.target_posx = self.assigned_vertex_pos[:,0]
        self.target_posy = self.assigned_vertex_pos[:,1]
        self.target_posz = self.assigned_vertex_pos[:,2]
        self.Path_Generation = path_generation()
        GlobalBest = self.Path_Generation.pso(self.xobs, self.yobs, self.zobs, self.robs, self.hobs, self.nObs, self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax, self.agent_posx, self.agent_posy, self.agent_posz, self.target_posx, self.target_posy, self.target_posz)
        ## Generated waypoints
        xx = GlobalBest.Sol.xx
        yy = GlobalBest.Sol.yy
        zz = GlobalBest.Sol.zz
        waypts_pso = np.zeros((self.num_agents,3))
        for n in range (self.look_ahead_num):    
            for j in range (self.num_agents):
                # get the next waypoint
                # C = self.rotation_from_axis_angle(axis[j], 0.02*angular_diff[j])
                # v = C@self.agent_vec[j]/norm(C@self.agent_vec[j])
                # waypoint = self.target_pos+(d_vertex_target[j]+0.98*radial_diff[j])*v
                waypoint = np.array([xx[1*n+1+j*100],yy[1*n+1+j*100],zz[1*n+1+j*100]])
                # clip the height to be between 0.1m and 2.0m
                # waypoint[-1] = np.clip(waypoint[-1], 0.1, 2.0)
                # populates look ahead points to be interpolated
                ts[n, j] = n*self.look_ahead_dt
                xs[n, j] = waypoint[0]
                ys[n, j] = waypoint[1]
                zs[n, j] = waypoint[2]
                if norm(waypoint - self.assigned_vertex_pos[j])<=0.05:
                    look_ahead_pts[n+1, j] = self.assigned_vertex_pos[j]
                else:
                    look_ahead_pts[n+1, j] = waypoint
        accx = np.zeros((self.num_agents, 1))
        accy = np.zeros((self.num_agents, 1))
        accz = np.zeros((self.num_agents, 1))
        downwash_acc = np.zeros((self.num_agents, 1))
        v_pso = np.zeros((self.num_agents,3))
        for j in range (self.num_agents):
            # interpolate for velocity and (accelerations --> roll, pitch)
            x_interp = UnivariateSpline(ts[:, j], xs[:, j],k=2)
            y_interp = UnivariateSpline(ts[:, j], ys[:, j],k=2)
            z_interp = UnivariateSpline(ts[:, j], zs[:, j],k=2)
            vx_interp = x_interp.derivative()
            vy_interp = y_interp.derivative()
            vz_interp = z_interp.derivative()
            ax_interp = vx_interp.derivative()
            ay_interp = vy_interp.derivative()
            az_interp = vz_interp.derivative()
            # pso generated accelns and vels
            waypt = look_ahead_pts[self.look_ahead_num//2, j] #[x,y,z]
            waypts_pso[j] = waypt
            vx_pso, vy_pso, vz_pso = vx_interp(self.look_ahead_num//2*self.look_ahead_dt), vy_interp(self.look_ahead_num//2*self.look_ahead_dt), vz_interp(self.look_ahead_num//2*self.look_ahead_dt)
            v_pso[j] = np.array([vx_pso, vy_pso, vz_pso])
            ax_pso, ay_pso, az_pso = ax_interp(self.look_ahead_num//2*self.look_ahead_dt), ay_interp(self.look_ahead_num//2*self.look_ahead_dt), az_interp(self.look_ahead_num//2*self.look_ahead_dt)
            print(f'{waypt} compared with {self.agent_pos[j,0]+vx_pso*self.look_ahead_dt+1/2*ax_pso*self.look_ahead_dt*self.look_ahead_dt} {self.agent_pos[j,1]+vy_pso*self.look_ahead_dt+1/2*ay_pso*self.look_ahead_dt*self.look_ahead_dt} {self.agent_pos[j,2]+vz_pso*self.look_ahead_dt+1/2*az_pso*self.look_ahead_dt*self.look_ahead_dt}')
            
            agent_input[0, j] = self.agent_ang[j,0] #deg !!!!!!
            agent_input[1, j] = self.agent_ang[j,1]  #deg !!!!!!!!

            if self.agent_h[j, 0] < 0.2:
                collision_acc[j, 2] = np.abs(collision_acc[j, 2])
            
            accx[j] = (max(min(ax_pso+collision_acc[j,0], Config.MaxAcc), -Config.MaxAcc))
            accy[j] = (max(min(ay_pso+collision_acc[j,1], Config.MaxAcc), -Config.MaxAcc))
            accz[j] = (max(min(ay_pso+collision_acc[j,2], Config.MaxAcc), -Config.MaxAcc))
            acc_xyz = [accx[j], accy[j], accz[j]]
            # acc = acc_xyz[2]
            downwash_acc[j] = self.get_dw_acc(downwash_flag[j], acc_xyz, neighbors_pos[j], agent_coord[j], agent_input[:2,j])
            # self.agent_h[j, 0] += self.agent_h[j, 1]*self.look_ahead_dt+(1/2)*acc*self.look_ahead_dt*self.look_ahead_dt-(1/2)*downwash_acc[j]*self.look_ahead_dt*self.look_ahead_dt 
            # self.agent_h[j, 1] += (acc-downwash_acc[j])*self.look_ahead_dt #IF NEED TO CONSIDER DW_ACC, THEN MIGHT NEED TO ADD DW_ACC ON TOP OF ACC
                
            # if downwash_acc[j] != 0.0:
            #     print("downwash_accel is",downwash_acc[j])
            #     print("agent no. is ",j)
            #     print("next agent")
        
        acc_total = np.column_stack((accx, accy, accz))   
        # print(f'acceleration is {acc_total}, accx is {accx}, accy is {accy}, accz is {accz}')
        vel = self.agent_vel + v_pso+ acc_total*self.look_ahead_dt
        vel[:,0] = np.maximum(np.minimum(vel[:,0], Config.MaxVelo), -Config.MaxVelo)
        vel[:,1] = np.maximum(np.minimum(vel[:,1], Config.MaxVelo), -Config.MaxVelo)
        vel[:,2] = np.maximum(np.minimum(vel[:,2], Config.MaxVelo), -Config.MaxVelo)
        for agent in range(self.num_agents):
            vel[agent,2] += downwash_acc[agent]*self.look_ahead_dt
        self.agent_pos_after = self.agent_pos + vel*self.look_ahead_dt
        print(f'self.agent_pos is {self.agent_pos} and velocity is {vel}')
        print(f'self.agent_pos_after is {self.agent_pos_after}')

        # self.vis(i=0)
        return self.agent_pos_after
        
    def get_dw_acc(self, dw_flag, acc_xyz, neighbors_pos, agent_coord, agent_input):
        multi = 10
        downwash_acc1 = 0.0
        DEG2RAD = np.pi/180
        pitch = agent_input[1]*DEG2RAD

        roll = agent_input[0]*DEG2RAD
        acc1 = acc_xyz
        # neighbors_pos = neighbors_pos
        # agent_coord = agent_coord
        # A_UAV = np.pi*((1/2)*0.092)**2
        A_UAV = np.pi*((1/2)*Config.droneSideLength)**2
        
        for k in range(len(dw_flag)):
            a_des = -np.sin(pitch)*acc1[0] + np.cos(pitch)*np.sin(roll)*acc1[1] + np.cos(pitch)*np.cos(roll)*(acc1[2]+9.81)
            z = neighbors_pos[k][2] - agent_coord[2]
            if dw_flag[k] == 1:
                print("delta z is ",z)
                print("a_des is", a_des)
                downwash_acc = 25 * A_UAV / 2 / np.pi * a_des / (z**2)
                downwash_acc1 += downwash_acc
                print("downwash_acc is ", downwash_acc)
                # print("dw_flag is", dw_flag[k])

        temp = multi*downwash_acc1
        return temp
        
    def vis(self,iter):
        # workbook = Workbook('4agents_1obs_static.xlsx')
        # Report_Sheet = workbook.add_worksheet()
        # Environment_Sheet = workbook.add_worksheet()
        # # Write the column headers if required.
        # Report_Sheet.write(0, 0, 'Transition Time(s)')
        # Environment_Sheet.write(0, 0, 'Starting Position x(m)')
        # Environment_Sheet.write(0, 1, 'Starting Position y(m)')
        # Environment_Sheet.write(0, 2, 'Starting Position z(m)')
        # Environment_Sheet.write(0, 3, 'Target Position x(m)')
        
        # # Write the column data.
        # Report_Sheet.write_column(1, 0, Column1)
        # Report_Sheet.write_column(1, 1, Column2)

        # workbook.close()
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.set_xlim3d(self.xmin, self.xmax, auto = False)
        # ax.set_xlim3d(-3, 3, auto = False)
        ax.set_ylim3d(self.ymin, self.ymax, auto = False)
        # ax.set_ylim3d(-2, 2, auto = False)
        ax.set_zlim3d(self.zmin, self.zmax, auto = False)
        # ax.set_zlim3d(0, 2, auto = False)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')

        ax.scatter3D(self.target_pos[0], self.target_pos[1], self.target_pos[2], color = 'red', label = 'target', marker = '*')
        ax.scatter3D(self.vertex_pos[:, 0], self.vertex_pos[:, 1], self.vertex_pos[:, 2], color = 'brown', label = 'vertices', marker = 'X')
            
        for agent in range (self.num_agents):
            ax.plot3D(self.agent_coords[agent, :(iter+1), 0], self.agent_coords[agent, :(iter+1), 1], self.agent_coords[agent, :(iter+1), 2], '-g', label = 'agent'+str(agent))
            # Report_Sheet.write(0, 3*agent+1, f'agent{agent+1}_x position(m)')
            # Report_Sheet.write(0, 3*agent+2, f'agent{agent+1}_y position(m)')
            # Report_Sheet.write(0, 3*agent+3, f'agent{agent+1}_z position(m)')
        for k in range(self.nObs):
            # axis and radius
            p1 = np.array([self.xobs[k], self.yobs[k], self.zobs[k]])
            # vector in direction of axis
            v = np.array([0,0,self.hobs[k]])
            p0 = p1-v
            R = self.robs[k]
            # find magnitude of vector
            mag = norm(v)
            # unit vector in direction of axis
            v = v/mag
            # make some vector not in the same direction as v
            not_v = np.array([1,0,0])
            if (v==not_v).all():
                not_v = np.array([0,1,0])
            # make unit vector perpendicular to v
            n1 = np.cross(v,not_v)
            # normalize n1
            n1 /= norm(n1)
            # make unit vector perpendicular to v and n1
            n2 = np.cross(v,n1)
            #surface ranges over t from 0 to length of axis and 0 to 2*pi
            t = np.linspace(0,mag,100)
            theta = np.linspace(0,2*np.pi,50) #divide the circle into 50 equal parts
            # use meshgrid to make 2d arrays
            t, theta = np.meshgrid(t,theta)
            # generate coordinates for surface
            X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
            ax.plot_surface(X, Y, Z)
            ax.plot(*zip(p0, p1), color = 'red')
        ax.legend()
        plt.show()
        fig.savefig("4agents_7movingobs",dpi = 300)


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    order = 1

    if order == 2:
        agent_init = np.zeros((8, 4)) 
    elif order == 1:
        # agent_init = np.zeros((6, 1)) #case 0 : single agent-single target-no obstacle
        agent_init = np.zeros((6, 4)) #case 1: 4 agents-single target-no obstacle

    # agent_init[:2, 0] = -1.5, 0.3
    # agent_init[:2, 1] = -1.5, 0.0
    # agent_init[:2, 2] = -1.5, -0.3
    # agent_init[:2, 3] = -1.5, -0.6

    # agent_init[:2, 0] = -1.0, 0.0
    # agent_init[:2, 1] = -1.2, 0.0
    # agent_init[:2, 2] = -1.4, 0.0
    # agent_init[:2, 3] = -1.6, 0.0

    # case1: 1 agent wo obstacle
    # agent_init[:2, 0] =  1.5, 1.5 
    # case2: 4 agent wo obstacle
    # agent_init[:2, 0] =  1.5, 1.5 
    # agent_init[:2, 1] = -1.5, 1.5
    # agent_init[:2, 2] =  1.5, -1.5
    # agent_init[:2, 3] = -1.0, -1.0
    # case3: 4 agent w/ 3 obstacles
    # agent_init[:2, 0] =  1.5, 1.5 
    # agent_init[:2, 1] = -1.5, 1.5
    # agent_init[:2, 2] =  1.5, -1.5
    # agent_init[:2, 3] = -1.0, -1.0
    # case4: 4 agent w/ 7 obstacles
    # agent_init[:2, 0] =  1.5, 1.5 
    # agent_init[:2, 1] = -1.5, 1.5
    # agent_init[:2, 2] =  1.5, -1.5
    # agent_init[:2, 3] = -1.0, -1.0
    # case5: 1 agent w 3 moving obstacle
    # agent_init[:2, 0] =  1.5, 1.5 
    # case6: 4 agent w/ 3 moving obstacles
    # agent_init[:2, 0] =  1.5, 1.5 
    # agent_init[:2, 1] = -1.5, 1.5
    # agent_init[:2, 2] =  1.5, -1.5
    # agent_init[:2, 3] = -1.0, -1.0
    # case7: 4 agent w/ 7 moving obstacles
    agent_init[:2, 0] =  1.5, 1.5
    agent_init[:2, 1] = -1.5, 1.5
    agent_init[:2, 2] =  1.5, -1.5
    agent_init[:2, 3] = -1.0, -1.0
    agent_pos = np.hstack((agent_init.T[:,:2],np.zeros((len(agent_init[0]),1))))
    agent_vel = np.zeros((len(agent_init[0]),3))
    agent_acc = np.zeros((len(agent_init[0]),3))
    # # case 1/2/3/4
    target_init = np.array([1.0, 0.5, 0.0, 0.0]).T
    target_pos = np.hstack((target_init.T[:2],1))
    agent_ang = np.zeros((4, 2))

    # case1/2
    # obs_init = np.zeros((7,1)) 
    # obs_init[:5,0] = -1.8, -1.8,    2, 0.3, 2 
    # case3
    # obs_init = np.zeros((7,3)) 
    # obs_init[:5,0] = -1.8, -1.8,    2, 0.3, 2 
    # obs_init[:5,1] =   -1,    0,  1.8, 0.1, 2
    # obs_init[:5,2] = -0.5,  1.5,    1, 0.5, 2
    # # case4
    # obs_init = np.zeros((7,7)) #[xobs, yobs, zobs, robs, hobs, vx, vy]*7
    # obs_init[:5,0] = -1.8, -1.8,    2, 0.3, 2 
    # obs_init[:5,1] =   -1,    0,  1.8, 0.1, 2
    # obs_init[:5,2] = -0.5,  1.5,    1, 0.5, 2
    # obs_init[:5,3] =    0, -0.5,  0.5, 0.2, 2
    # obs_init[:5,4] =  0.5,  1.5,  1.5, 0.4, 2
    # obs_init[:5,5] =  1.7, -0.5,  1.3, 0.5, 2
    # obs_init[:5,6] =    0,   -2,    2, 0.5, 2
    # case5
    # obs_init = np.zeros((7,3)) 
    # obs_init[:7,0] = -1.8, -1.8,    2, 0.3, 2 , 0.1, 0.2
    # obs_init[:7,1] =   -1,    0,  1.8, 0.1, 2 , 0.1, -0.1
    # obs_init[:7,2] = -0.5,  1.5,    1, 0.5, 2 , 0.1, -0.05  
    # case6
    # obs_init = np.zeros((7,3)) 
    # obs_init[:7,0] = -1.8, -1.8,    2, 0.3, 2 , 0.1, 0.2
    # obs_init[:7,1] =   -1,    0,  1.8, 0.1, 2 , 0.1, -0.1
    # obs_init[:7,2] = -0.5,  1.5,    1, 0.5, 2 , 0.1, -0.5 
    # # case4
    # obs_init = np.zeros((7,7)) #[xobs, yobs, zobs, robs, hobs, vx, vy]*7
    # obs_init[:7,0] = -1.8, -1.8,    2, 0.3, 2 , 0.1, 0.2
    # obs_init[:7,1] =   -1,    0,  1.8, 0.1, 2 , 0.1, -0.1
    # obs_init[:7,2] = -0.5,  1.5,    1, 0.5, 2 , 0.1, -0.5
    # obs_init[:7,3] =    0, -0.5,  0.5, 0.2, 2 , 0.1, 0.2
    # obs_init[:7,4] =  0.5,  1.5,  1.5, 0.4, 2 , 0.1, -0.1
    # obs_init[:7,5] =  1.7, -0.5,  1.3, 0.5, 2 , 0.1, -0.5
    # obs_init[:7,6] =    0,   -2,    2, 0.5, 2 , 0.1, -0.1
   
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    zmin, zmax = 0, 2
    
    # sim_2D = Sim2D(agent_init, target_init, obs_init, num_iter = 200, dt = 0.5, order = order)
    sim_2D = Sim2D(agent_pos, agent_vel, agent_acc, agent_ang, target_pos, obs_init=None, dt = 0.5)
    # sim_2D = Sim2D(agent_init, target_init, obs_init, num_iter = 1700, dt = 0.01, order = order)
    start = time.time()
    sim_2D.run(Config.l, Config.look_ahead_num, Config.look_ahead_dt)
    end = time.time()
    print(f"Runtime of the program is {end - start}")