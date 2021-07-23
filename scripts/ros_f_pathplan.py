#!/usr/bin/env python

'''
Create a package called ros_pathplan
Git clone the ORCA algorithm (Python-RVO2-3D)
and append path to the folder
rosrun ros_pathplan ros_pathplan.py
'''

# import matplotlib.pyplot as plt
# import matplotlib.patches as ptch
# import matplotlib
# from matplotlib import cm
import numpy as np
# import math
# import time
# import csv
# from termcolor import colored

import sys  
from Config import Config
import agent
import flocking
import downwash

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from outdoor_gcs.msg import PathPlan


class F_PathPlan(object):

    def __init__(self, dt):

        print("FLOCK!!")

        self.dt = dt
        self.change_time = rospy.Time.now()

        self.agents_num, self.max_agents_num = 0, 9
        self.cur_pos = np.zeros((3*self.max_agents_num)) # max 5 drones, each with xyz
        self.cur_vel = np.zeros((3*self.max_agents_num))
        self.des_pos = np.zeros((3*self.max_agents_num))

        self.start_sim = False
        self.uavs_id = [False,False,False,False,False,False,False,False,False]

        self.agents = []
        self.c1, self.c2, self.RG = 10.0, 10.0, 10000.0
        self.r_alpha, self.MaxAcc, self.MaxVelo = 1.0, 10.0, 10.0 

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
        self.pathplan_pub.publish(self.pathplan_pub_msg_nxt)

    def iteration(self, event):
        if (self.start_sim and rospy.Time.now()-self.change_time > rospy.Duration(secs=5)):
            self.change_time = rospy.Time.now()
            self.pathplan_pub_msg_nxt.start, self.start_sim = False, False
            for i in range(self.max_agents_num):
                if (self.uavs_id[i] and len(self.agents) != self.agents_num):
                    self.agents.append(agent.Agent(self.cur_pos[i*3], self.cur_pos[i*3+1], self.cur_pos[i*3+2],
                        self.des_pos[i*3], self.des_pos[i*3+1], self.des_pos[i*3+2], i))
        if (self.agents_num == len(self.agents)):
            self.Move_One_Step()
            # self.update_nxtpos()
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

      rospy.init_node('ros_f_pathplan', anonymous=True)
      dt = 1.0/15*3
      pathplan_run = F_PathPlan(dt)
      rospy.Timer(rospy.Duration(dt), pathplan_run.iteration)
      rospy.spin()


     


