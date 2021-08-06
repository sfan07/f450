import numpy as np
import agent
from Config import Config

def downwash_force(host, agents):
    f = np.array([0.0, 0.0, 0.0])
    roll_ddot, pitch_ddot = 0.0, 0.0
    for other_agent in agents:
        if other_agent.id == host.id:
            continue
        dist_v = host.pos_global_frame - other_agent.pos_global_frame
        # print(host.id, host.pos_global_frame)
        # print(other_agent.id, other_agent.pos_global_frame)
        # Other agent on top, host is below
        if Config.DW_TILT_CYL:
            M = other_agent.R_BI
            dist_v2 = np.dot(np.transpose(M), dist_v) # in neighbor body frame
            d = (dist_v2[0]**2+dist_v2[1]**2)**0.5
            if (d < 2*host.radius) and (dist_v2[2] > 0): # host is below
                A_uav = host.radius**2*np.pi # whole surface area of uav
                # A_uav = 2*host.radius**2*np.arccos(d/(2*host.radius)) - d*np.sqrt(host.radius**2 - (d/2)**2) # intersection area of 2 uavs
                
                para = 25.0*A_uav*other_agent.thrust_m/(2*np.pi*(dist_v2[2]**2))
                f = -M[:,2]*para

                if Config.TORQUE_FLIP:
                    M2 = host.R_BI
                    torque = np.dot(M2[:,2], -f)
                    alpha = torque*(d/2)/(host.radius**2/4) # angular acceleration

                    beta = np.arctan2(np.abs(dist_v2[0]), np.abs(dist_v2[1]))
                    if beta <= 10*np.pi/180:    roll_ddot = alpha
                    elif beta >= 80*np.pi/180:  pitch_ddot = alpha 
                    else: roll_ddot, pitch_ddot = alpha/np.sqrt(2), alpha/np.sqrt(2) 

                    if dist_v2[0] < 0: pitch_ddot = -pitch_ddot
                    if dist_v2[1] > 0: roll_ddot = -roll_ddot
        else:
            if (dist_v[0]**2+dist_v[1]**2)**0.5 < 2*host.radius and dist_v[2] < 0:
                f[2] = 25.0*(host.radius**2*np.pi)*other_agent.thrust_m/(2*np.pi*(dist_v[2]**2))
                # if host.id == 1:
                # print(host.id, other_agent.id, (dist_v[0]**2+dist_v[1]**2)**0.5, 2*host.radius, f)
    #     data_list2.append([t, -f[2], roll_ddot, pitch_ddot])
    # print(host.id, f)
    return(-f)
    # host.update_dw(-f, roll_ddot, pitch_ddot)