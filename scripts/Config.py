import os
import re
import numpy as np
import time
from termcolor import colored


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

