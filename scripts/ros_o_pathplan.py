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
sys.path.append('/home/chihunen/catkin_ws/src/ros_pathplan/src/Python-RVO2-3D') 
import rvo23d

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from outdoor_gcs.msg import PathPlan


class O_PathPlan(object):

    def __init__(self, dt):

        self.dt = dt
        self.change_time = rospy.Time.now()

        self.cur_pos = np.zeros((15)) # max 5 drones, each with xyz
        self.des_pos = np.zeros((15))

        self.start_sim = False
        self.uavs_id = [False,False,False,False,False]
        self.agents_num, self.max_agents_num = 0, 5
        self.radius, self.neighborDist = 1.0, 3.0
        self.timeHorizon, self.MaxVelo, self.velocity = 5.0, 10.0, (0,0,0)

        self.sim = rvo23d.PyRVOSimulator(self.dt, self.neighborDist, self.max_agents_num,
                        self.timeHorizon, self.radius, self.MaxVelo, self.velocity)

        self.pathplan_pub_msg, self.pathplan_pub_msg_nxt = PathPlan(), PathPlan()
        rospy.Subscriber("/uavs/pathplan", PathPlan, self.pathplan_callback)
        self.pathplan_pub = rospy.Publisher("/uavs/pathplan_nxt", PathPlan, queue_size=1)


    def update_nxtpos(self):
        self.pathplan_pub_msg_nxt.header.stamp = rospy.Time.now()
        nxt, j = [], 0
        for i in range(self.max_agents_num):
            if (self.uavs_id[i]):
                nxt.extend([self.sim.getAgentPosition(j)[0], self.sim.getAgentPosition(j)[1], self.sim.getAgentPosition(j)[2]])
                # print([self.sim.getAgentPosition(j)[0], self.sim.getAgentPosition(j)[1], self.sim.getAgentPosition(j)[2]])
                j+=1
            else:
                nxt.extend([0,0,0])
        self.pathplan_pub_msg_nxt.nxt_position = list(nxt)
        # print(self.pathplan_pub_msg.nxt_position)

    def publish_msg(self):
        self.pathplan_pub.publish(self.pathplan_pub_msg_nxt)

    def iteration(self, event):
        if (self.start_sim and rospy.Time.now()-self.change_time > rospy.Duration(secs=5)):
            self.change_time = rospy.Time.now()
            self.pathplan_pub_msg_nxt.start, self.start_sim = False, False
            for i in range(self.max_agents_num):
                if (self.uavs_id[i] and self.sim.getNumAgents()!=self.agents_num):
                    A = self.sim.addAgent((self.cur_pos[i*3], self.cur_pos[i*3+1], self.cur_pos[i*3+2]), 
                        self.neighborDist, self.max_agents_num, self.timeHorizon, self.radius, self.MaxVelo, self.velocity)
        if (self.agents_num != 0):
            j = 0
            # print(self.sim.getNumAgents())
            for i in range(self.max_agents_num):
                prefV = np.zeros((3))
                if (self.uavs_id[i]):
                    self.update_params(j)
                    self.sim.setAgentPosition(j, (self.cur_pos[i*3], self.cur_pos[i*3+1], self.cur_pos[i*3+2]))
                    vel = self.des_pos[i*3:i*3+3] - self.cur_pos[i*3:i*3+3]
                    nor = np.linalg.norm(vel)
                    if nor < 10**-4:
                        prefV[0], prefV[1], prefV[2] = 0.0, 0.0, 0.0
                    elif nor < 0.5:
                        prefV = vel
                    else:
                        prefV = vel/nor*self.MaxVelo
                    self.sim.setAgentPrefVelocity(j, (prefV[0], prefV[1], prefV[2]))
                    j+=1
            self.sim.doStep()
            self.update_nxtpos()
            self.publish_msg()


    
    def pathplan_callback(self, msg):
        self.pathplan_pub_msg = msg
        self.start_sim = msg.start
        self.agents_num = msg.num
        self.uavs_id = msg.uavs_id
        self.cur_pos = np.asarray(msg.cur_position)
        self.des_pos = np.asarray(msg.des_position)
        self.timeHorizon = msg.params[0]
        self.MaxVelo = msg.params[1]
        self.radius = msg.params[2]
        self.neighborDist = msg.params[3]
    
    def update_params(self, j):
        if (self.sim.getAgentTimeHorizon(j) != self.timeHorizon):
            self.sim.setAgentTimeHorizon(j, self.timeHorizon)
        if (self.sim.getAgentMaxSpeed(j) != self.MaxVelo):
            self.sim.setAgentMaxSpeed(j, self.MaxVelo)
        if (self.sim.getAgentRadius(j) != self.radius):
            self.sim.setAgentRadius(j, self.radius)
        if (self.sim.getAgentNeighborDist(j) != self.neighborDist):
            self.sim.setAgentNeighborDist(j, self.neighborDist)


if __name__ == '__main__':

      rospy.init_node('ros_o_pathplan', anonymous=True)
      dt = 1.0/15
      pathplan_run = O_PathPlan(dt)
      rospy.Timer(rospy.Duration(dt), pathplan_run.iteration)
      rospy.spin()


     


