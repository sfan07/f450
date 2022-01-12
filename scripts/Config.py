import os
import re
import numpy as np
import time
# from termcolor import colored


class Config:

    ALG_MODE = 'Flocking'
    File_name = '/test_1'
    Data_file = File_name + '.csv'
    env_dimention = 3 # Don't change this

    radius, height, MaxVelo, MaxAcc = 0.47/(2**0.5), 0.11, 50.0, 120.0 #note: height here is the cylinder height
    # radius, height, MaxVelo, MaxAcc = 0.40, 0.50, 3.0, 5.0
    r_alpha, h_alpha = radius*3, height/2*3 #1.5 #height*3/2*6height*3/2 #0.57 0.74 @a=5
    r_alpha, h_alpha = 1.0, 2.5 #1.5 #height*3/2*6height*3/2 #0.57 0.74 @a=5
    r_alpha_max, r_alpha_min = r_alpha, r_alpha*1/2
    h_alpha_max, h_alpha_min = 1.16, 0.2
    # r_star = radius*1.0 # 3.0*radius # was 3
    # h_star = 2.0*radius # 0, 0.25, 0.5, 1.0, 1.25, 1.5, 2 
    # d = 2*host.radius + Config.d_star # + 0.3
    # r_alpha = d #np.cbrt(3*np.square(host.pref_speed)/(2*RepulsiveGradient)) + d
    # # h = 2*host.height + Config.h_star
    # h_alpha = Config.h_star #np.cbrt(3*np.square(host.pref_speed)/(2*RepulsiveGradient)) + h
    # MaxSimTime, DeltaT, EndMaxDis = 25.0, 0.01, 0.05
    z_min = 0.0

    c1, c2, RepulsiveGradient = 10.0, 10.0, 10000.0 #*(10**3)
    MaxVelo, MaxAcc = 10.0, 10.0

    world = np.array([[0,1,0], [1,0,0], [0,0,-1]])
    world = np.array([[1,0,0], [0,1,0], [0,0,1]])
    # videocut = 5
    # init_gap = radius*3.0
    # Loop = 1

    '''Dynamics'''
    DYN = True
    DYN = False

    '''Bump Function'''
    BUMP = True
    BUMP = False
    # bump_c = RepulsiveGradient
    h_bump = 0.5

    '''Repulsive force cylinder model?'''
    CYL_MODEL = True
    CYL_MODEL = False   
    '''Adaptive height'''
    ADAPT_H = True
    ADAPT_H = False
    '''Adaptive model (height+radius)'''
    ADAPT_RH = True
    ADAPT_RH = False

    '''Plot drones? for plots_3D.py'''
    PLOT_DRONE = True
    # PLOT_DRONE = False
    PLOT_CYL = True
    PLOT_CYL = False
    PLOT_CYL_R = radius #r_alpha #radius
    PLOT_CYL_H = h_alpha #radius*4.0
    plot_num = 0 # number of drones image per drone

    '''Test dw force?'''
    TEST_DW_F = True
    # TEST_DW_F = False

    '''Test tilted dw model?'''
    DW_TILT_CYL = True
    DW_TILT_CYL = False

    '''Test torque produced by dw?'''
    TORQUE_FLIP = True
    TORQUE_FLIP = False

    '''Test zero navigational feedback?'''
    ZERO_NAV_F = True
    ZERO_NAV_F = False

    '''distance between vertices and the assigned target position'''
    l = 1
    look_ahead_num = 3
    # look_ahead_dt = 0.1
    look_ahead_dt = 1.0/15*1/2

    '''Drone dynamics'''
    droneSideLength = 0.33*2

    '''pso parameters'''
    MaxIt_pso = 20 # Maximum Number of Iterations
    nPop_pso = 20 # population size (swarm size)
    # nPop_pso = 100 # population size (swarm size)
    w_pso = 1 # inertia weight
    wdamp_pso = 0.98 # inertia weight damping ratio
    c1_pso = 1.5 # personal learning coefficient
    c2_pso = 2 # Global Learning Coefficient
    alpha_pso = 0.1 #adjusting velocity limits
    beta = 500 #z = sol.L*(1+Config.beta*sol.Violation)

    '''accl limits and vel limits'''
    MaxAcc, MaxVelo = 10.0, 10.0 

    '''adjust this number to influence hunting speed, must larger than 20'''
    # Seperate = 20
    # Seperate = 30
    # Seperate = 10
    # Seperate = 7
    # Seperate = 4
    Seperate = 6
    # Seperate = 15

    '''adjust downwash acceleration, multi*dw_acc'''
    multi = 10

    # '''One against one'''
    # agentNo = 1

    '''Four against one'''
    agentNo = 4

    '''None obstacle case'''
    obs_init = None #[x,y,z,r,h,vx,vy].T

    # '''one obstacle case'''
    # obs_init = np.zeros((7,1)) #[x,y,z,r,h,vx,vy].T
    # obs_init[:5,0] = -1.8, -1.8,    20, 0.5, 20 

    # '''three obstacles case'''
    # obs_init = np.zeros((7,3)) #[x,y,z,r,h,vx,vy].T
    # obs_init[:5,0] = -1.8, -1.8,    20, 0.5, 20 
    # obs_init[:5,1] = 1.8, 1.8,    20, 0.5, 20 
    # obs_init[:5,2] = 1.8, -1.8,    20, 0.5, 20 

 # Parameters for the path planner from Jacky's path planner algorithm
# planner:
    x_min=-10        # float, minimum x value in m for the planner
    x_max=10         # float, maximum x value in m for the planner
    y_min=-10        # float, minimum y value in m for the planner
    y_max=10         # float, maximum y value in m for the planner
    z_min= 0         # float, minimum z value in m for the planner
    z_max=10         # float, maximum z value in m for the planner
    dx=0.1       # float, planner discretization step in x
    dy=0.1       # float, planner discretization step in y
    dz=0.1       # float, planner discretization step in z
    dt=0.1       # float, planner discretization step in t
    horizon=25        # float, amount of time to plan ahead in s
    window=5         # float, window size/amount of time to consider other agents
    j_t=0.2       # float, cost of stepping in time
    j_s=1         # float, cost of stepping in space
    obs_padding=0.0       # float, padding distance to add to obstacle radius
    d_replan=0.25      # float, distance error of target/obstacle to trigger replanning
    v_replan=0.1       # float, velocity error of target/obstacle to trigger replanning
    t_replan =6         # float, time difference from t_max to trigger replanning   