#!/usr/bin/env python3

'''
Create a package called ros_pathplan
Git clone the ORCA algorithm (Python-RVO2-3D)
# and append path to the folder
rosrun ros_pathplan ros_pathplan.py
'''

# import matplotlib.pyplot as plt
# import matplotlib.patches as ptch
# import matplotlib
# from matplotlib import cm
import time
import numpy as np
# import math
# import time
# import csv
# from termcolor import colored

import sys  
from Config import Config
# import agent
# import flocking
# import downwash

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from outdoor_gcs.msg import PathPlan
import sim_3D_static_target as sim_3D_static_target
# import sim_3D_static_target_s as sim_3D_static_target


class PSO_PathPlan(object):

    def __init__(self, dt):

        print("jacky!!")
        

        self.dt = dt
        self.change_time = rospy.Time.now()

        self.agents_num, self.max_agents_num = 0, 9
        self.cur_pos = np.zeros((3*self.max_agents_num)) # max 5 drones, each with xyz
        self.cur_vel = np.zeros((3*self.max_agents_num))
        self.des_pos = np.zeros((3*self.max_agents_num))
        self.cur_ang = np.zeros((3*self.max_agents_num))
        self.cur_acc = np.zeros((3*self.max_agents_num))


        self.start_sim = False
        self.uavs_id = [False,False,False,False,False,False,False,False,False]

        self.agents = []
        self.agent_pos = []
        self.agent_vel = []
        self.agent_acc = []
        self.target_pos = []
        self.agent_ang = []
        self.c1, self.c2, self.RG = 10.0, 10.0, 10000.0
        self.r_alpha, self.MaxAcc, self.MaxVelo = 1.0, Config.MaxAcc, Config.MaxVelo 

        self.pathplan_pub_msg, self.pathplan_pub_msg_nxt = PathPlan(), PathPlan()
        rospy.Subscriber("/uavs/pathplan", PathPlan, self.pathplan_callback)
        self.pathplan_pub = rospy.Publisher("/uavs/pathplan_nxt", PathPlan, queue_size=1)

    def pathplan_callback(self, msg):
        self.pathplan_pub_msg = msg
        self.start_sim = msg.start
        self.agents_num = msg.num
        self.uavs_id = msg.uavs_id
        self.cur_pos = np.asarray(msg.cur_position)
        self.cur_vel = np.asarray(msg.cur_velocity)
        
        self.des_pos = np.asarray(msg.des_position)
        # print(f'received back the pathplan msg, des_pos is {self.des_pos} ')
        self.cur_acc = np.asarray(msg.cur_acceleration)
        self.cur_ang = np.asarray(msg.cur_angles)
        
        if (msg.params[0]!= 0.0): self.c1 = msg.params[0]
        if (msg.params[1]!= 0.0): self.c2 = msg.params[1]
        if (msg.params[2]!= 0.0): self.RG = msg.params[2]
        if (msg.params[3]!= 0.0): self.r_alpha = msg.params[3]
        if (msg.params[4]!= 0.0): self.MaxAcc = msg.params[4]
        if (msg.params[5]!= 0.0): self.MaxVelo = msg.params[5]
        # self.c1, self.c2, self.RG = msg.params[0], msg.params[1], msg.params[2]
        # self.r_alpha, self.MaxAcc, self.MaxVelo = msg.params[3], msg.params[4], msg.params[5]

    def update_nxtpos(self):
        self.pathplan_pub_msg_nxt.header.stamp = rospy.Time.now()
        nxt, j = [], 0
        for i in range(self.max_agents_num):
            if (self.uavs_id[i]):
                pos = self.agents[j].pos_global_frame
                nxt.extend([pos[0], pos[1], pos[2]])
                # print([self.sim.getAgentPosition(j)[0], self.sim.getAgentPosition(j)[1], self.sim.getAgentPosition(j)[2]])
                j+=1
            else:
                nxt.extend([0,0,0])
        self.pathplan_pub_msg_nxt.nxt_position = list(nxt)
        # print(self.pathplan_pub_msg.nxt_position)

    def publish_msg(self):
        self.pathplan_pub_msg_nxt.header.stamp = rospy.Time.now()
        print('in')
        self.pathplan_pub.publish(self.pathplan_pub_msg_nxt)

    def iteration(self, event):
        # #obstacle case testing
        # self.obs_init = np.zeros((7,1)) #[x,y,z,r,h,vx,vy].T
        # self.obs_init[:5,0] = -1.8, -1.8,    20, 0.3, 20  
        self.obs_init = Config.obs_init
        # if self.obs_init.any() != None:
        #     print("with obstacle!!!")
        self.agent_pos = []
        self.agent_vel = []
        self.agent_acc = []
        self.target_pos = []
        self.agent_ang = []
        if (self.start_sim):
            # self.pathplan_pub_msg_nxt.start = True
            j = 0
            for i in range(self.max_agents_num):
                if (self.uavs_id[i] and len(self.agent_pos) != self.agents_num):
                    if j == 0:
                        self.target_pos.append(np.array([self.cur_pos[i*3], self.cur_pos[i*3+1], self.cur_pos[i*3+2]],dtype='float64'))
                        print(self.target_pos)
                        j += 1
                    else:
                        self.agent_pos.append(np.array([self.cur_pos[i*3], self.cur_pos[i*3+1], self.cur_pos[i*3+2]],dtype='float64'))
                        self.agent_vel.append(np.array([self.cur_vel[i*3], self.cur_vel[i*3+1], self.cur_vel[i*3+2]],dtype='float64'))
                        self.agent_acc.append(np.array([self.cur_acc[i*3], self.cur_acc[i*3+1], self.cur_acc[i*3+2]],dtype='float64'))
                        self.agent_ang.append(np.array([self.cur_ang[i*3], self.cur_ang[i*3+1], self.cur_ang[i*3+2]],dtype='float64'))
            nxt = []
            assigned_vertices = []
            
            self.agent_pos = np.array(self.agent_pos).reshape(len(self.agent_pos),3)
            self.target_pos = np.array(self.target_pos).reshape(1,3)
            self.agent_vel = np.array(self.agent_vel).reshape(len(self.agent_pos),3)
            self.agent_acc = np.array(self.agent_acc).reshape(len(self.agent_pos),3)
            self.agent_ang = np.array(self.agent_ang).reshape(len(self.agent_pos),3)
            # self.sim = sim_3D_static_target.Sim2D(self.agent_pos, self.agent_vel, self.agent_acc, self.agent_ang, self.target_pos, obs_init=None, dt = 0.5)
            self.sim = sim_3D_static_target.Sim2D(self.agent_pos, self.agent_vel, self.agent_acc, self.agent_ang, self.target_pos, self.obs_init, dt = 0.5)
            agent_nxt, agent_assigned_vertices = self.sim.run(Config.l, Config.look_ahead_num, Config.look_ahead_dt)

            targetskipped = False
            
            for agent in range(self.max_agents_num):
                # print(f'next waypoint agent_nxt[0] is {agent_nxt[0]}')
                if (self.uavs_id[agent]):
                    if (targetskipped == False):
                        targetskipped = True
                        j = 0
                        nxt.extend([self.des_pos[agent*3], self.des_pos[agent*3+1], self.des_pos[agent*3+2]])
                        assigned_vertices.extend([self.des_pos[agent*3], self.des_pos[agent*3+1], self.des_pos[agent*3+2]])
                        continue
                    else:
                        nxt.extend(agent_nxt[j].tolist())
                        assigned_vertices.extend(agent_assigned_vertices[j].tolist())
                        j += 1
                else:
                    nxt.extend([0,0,0])
                    assigned_vertices.extend([0,0,0])
                        
            self.pathplan_pub_msg_nxt.nxt_position = list(nxt)
            self.pathplan_pub_msg_nxt.des_position = list(assigned_vertices)
            self.publish_msg()

    def Move_One_Step(self):
        forces, nxt, j = [], [], 0
        rx_mat, ry_mat, height_mat = [], [], []
        for i in range(self.max_agents_num):
            rx_mat.append({})
            ry_mat.append({})
            height_mat.append({})
        for i in range(self.max_agents_num):
            if (self.uavs_id[i]):
                self.agents[j].update_info( self.cur_pos[i*3], self.cur_pos[i*3+1], self.cur_pos[i*3+2],
                                            self.cur_vel[i*3], self.cur_vel[i*3+1], self.cur_vel[i*3+2],
                                            self.des_pos[i*3], self.des_pos[i*3+1], self.des_pos[i*3+2])
                host_agent =  self.agents[j]
                force = flocking.move(host_agent, self.agents, self.c1, self.c2, self.RG, self.r_alpha, rx_mat, ry_mat, height_mat)
                # print(host_agent.id, force)
                if Config.TEST_DW_F:
                    # downwash.downwash_force(host_agent, self.agents)
                    dw_f = downwash.downwash_force(host_agent, self.agents)

                # self.agents[j].update_state(force, self.dt, self.MaxAcc, self.MaxVelo)
                j+=1
                # print(force, dw_f)

                force[0] = max(min(force[0], self.MaxAcc), -self.MaxAcc)
                force[1] = max(min(force[1], self.MaxAcc), -self.MaxAcc)
                force[2] = max(min(force[2], self.MaxAcc), -self.MaxAcc)
                vel = self.cur_vel[i*3:i*3+3] + force * self.dt
                vel[0] = max(min(vel[0], self.MaxVelo), -self.MaxVelo)
                vel[1] = max(min(vel[1], self.MaxVelo), -self.MaxVelo)
                vel[2] = max(min(vel[2], self.MaxVelo), -self.MaxVelo)
                vel += dw_f*self.dt
                nxt.extend(self.cur_pos[i*3:i*3+3] + vel*self.dt)
            else:
                nxt.extend([0,0,0])
        self.pathplan_pub_msg_nxt.nxt_position = list(nxt)


if __name__ == '__main__':

      rospy.init_node('ros_pso_pathplan', anonymous=True)
      dt = 1.0/15*3
      pathplan_run = PSO_PathPlan(dt)
      rospy.Timer(rospy.Duration(dt), pathplan_run.iteration)
      rospy.spin()


     


