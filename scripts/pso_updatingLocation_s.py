#!/usr/bin/python3
import numpy as np
from numpy.core.defchararray import title
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation
from numpy.linalg import norm
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize
from scipy.interpolate import UnivariateSpline
import math
from vpython import cylinder, vector
from scipy.interpolate import interp1d
import numpy.matlib
import random
import copy
import timeit
import time
from numba import njit, prange
from Config import Config

class model():
    def __init__(self):
        self.xobs = []
        self.yobs = []
        self.zobs = []
        self.robs = []
        self.hobs = []
        self.nobs = []
        self.n = []
        self.xmin = []
        self.xmax = []
        self.ymin = []
        self.ymax = []
        self.zmin = []
        self.zmax = []
        self.obstBuffer = []
        self.nUAVs = 0

    def update_param(self, xobs, yobs, zobs, robs, hobs, nobs, n, xmin, xmax, ymin, ymax, zmin, zmax, obstBuffer, xs, ys, zs, xt, yt, zt):
        self.xobs = xobs
        self.yobs = yobs
        self.zobs = zobs
        self.robs = robs
        self.hobs = hobs
        self.nobs = nobs
        self.n = n
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.obstBuffer = obstBuffer
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.xt = xt
        self.yt = yt
        self.zt = zt
        self.nUAVs = len(xs)

class Position():
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []

class Velocity():
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []     

class Best():
    def __init__(self):
        self.Position = Position()
        self.Velocity = Velocity()
        # self.Cost = math.inf
        self.Cost = np.Inf
        # self.Cost = np.full((Config.agentNo,1), np.inf) # agent1:[0], agent nth : [n-1]
        self.PathLength = []
        self.Sol = sol2()

class empty_particle():
    def __init__(self):
        self.Position = Position()
        self.Velocity = Velocity()
        self.Cost = []
        self.PathLength = []
        self.Sol = sol2()
        self.Best = Best()

class GlobalBest():
    def __init__(self):
        # self.Cost = math.inf
        self.Cost = np.Inf 
        # self.Cost = np.full((Config.agentNo,1), np.inf) # agent1:[0], agent nth : [n-1] #check, not used
        self.PathLength = [] #check, not used
        self.Best = Best()
        self.Position = Position() #check, not used
        self.Sol = sol2() #check, not used

class sol2():
    def __init__(self):
        self.TS = []
        self.XS = []
        self.YS = []
        self.ZS = []
        self.tt = []
        self.xx = []
        self.yy = []
        self.zz = []
        self.dx = []
        self.dy = []
        self.dz = []
        self.L = [] 
        self.Violation = []
        self.IsFeasible = (self.Violation==0)

    def update_param(self, TS, XS, YS, ZS, tt, xx, yy, zz, L, L_each, Violation, Violation_each):
        self.TS = TS
        self.XS = XS
        self.YS = YS
        self.ZS = ZS
        self.tt = tt
        self.xx = xx
        self.yy = yy
        self.zz = zz
        # self.dx = dx
        # self.dy = dy
        # self.dz = dz
        self.L = L 
        self.L_each = L_each
        self.Violation = Violation
        self.Violation_each = Violation_each
        self.IsFeasible = (self.Violation_each==0)

class path_generation():
    '''
    input: 
    obstacles profiles [x,y,z,h,r]
    agent positions [xs,ys,zs]
    target position [xt,yt,zt]
    look_ahead_num: constant number

    output: 
    waypoints [x,y,z]
    '''

    def __init__(self):
        self.model = model()
        #self.particle = empty_particle()
        self.empty_particle = empty_particle()
        self.VarMin = Position()
        self.VarMax = Position()
        self.VelMax = Velocity()
        self.VelMin = Velocity()
        self.sol1 = Position()
        self.sol2 = sol2()
        self.GlobalBest = GlobalBest()
        self.temp_particle = empty_particle()

    def pso(self, xobs, yobs, zobs, robs, hobs, nObs, xmin, xmax, ymin, ymax, zmin, zmax, xs, ys, zs, xt, yt, zt):
        '''
        This function generates path waypoints for agents
        '''
        # droneSideLenght = 0.15
        # droneSideLenght = 0 # point
        droneSideLength = Config.droneSideLength # point
        # obstBuffer = droneSideLenght*1.5
        obstBuffer = droneSideLength*2
        # Number of intermediate way points
        # n = max(math.ceil(nObs/5)*3, 3)
        n = 3

        self.model.update_param(xobs, yobs, zobs, robs, hobs, nObs, n, xmin, xmax, ymin, ymax, zmin, zmax, obstBuffer, xs, ys, zs, xt, yt, zt)
        nVar = self.model.n # Number of Decision Variables
        
        VarSize = [1, nVar] # Size of Decision Variable Matrix

        self.VarMin.x = self.model.xmin # Lower Bound of Variables
        self.VarMax.x = self.model.xmax # Upper Bound of Variables
        self.VarMin.y = self.model.ymin
        self.VarMax.y = self.model.ymax
        self.VarMin.z = self.model.zmin
        self.VarMax.z = self.model.zmax

        '''
        PSO Parameters
        '''
        self.MaxIt = Config.MaxIt_pso # Maximum Number of Iterations
        # time = np.zeros((1,self.MaxIt))
        nPop = Config.nPop_pso # population size (swarm size)
        # nPop = 150 # population size (swarm size)
        w = Config.w_pso # inertia weight
        wdamp = Config.wdamp_pso # inertia weight damping ratio
        c1 = Config.c1_pso # personal learning coefficient
        c2 = Config.c2_pso # Global Learning Coefficient

        alpha = Config.alpha_pso #adjusting velocity limits
        # self.VelMax.x = alpha*(self.VarMax.x-self.VarMin.x) # Maximum Velocity
        # self.VelMin.x = -self.VelMax.x                # Minimum Velocity
        # self.VelMax.y =  alpha*(self.VarMax.y-self.VarMin.y) # Maximum Velocity
        # self.VelMin.y = -self.VelMax.y                # Minimum Velocity
        # self.VelMax.z =  alpha*(self.VarMax.z-self.VarMin.z) # Maximum Velocity
        # self.VelMin.z = -self.VelMax.z                # Minimum Velocity        
        self.VelMax.x = Config.MaxVelo                  # Maximum Velocity
        self.VelMin.x = -Config.MaxVelo                 # Minimum Velocity
        self.VelMax.y = Config.MaxVelo                  # Maximum Velocity
        self.VelMin.y = -Config.MaxVelo                 # Minimum Velocity
        self.VelMax.z = Config.MaxVelo                  # Maximum Velocity
        self.VelMin.z = -Config.MaxVelo                 # Minimum Velocity
        
        # Initialize Global Best
        self.GlobalBest.Best.Cost = np.Inf
        # self.GlobalBest.Best.Cost = np.full((Config.agentNo,1), np.inf) # agent1:[0], agent nth : [n-1]

        # Create Particles Matrix
        self.particle = {} #empty_particle()
        # Initialization Position
        xyzs = np.stack((self.model.xs, self.model.ys, self.model.zs))
        xyzt = np.stack((self.model.xt, self.model.yt, self.model.zt))
        dist_xyz = np.abs(xyzs-xyzt)
        sigma = dist_xyz/(self.model.n+1)/2

        sigma_x = np.matmul(sigma[0].reshape((self.model.nUAVs,1)),np.ones((1,self.model.n)))
        sigma_y = np.matmul(sigma[1].reshape((self.model.nUAVs,1)),np.ones((1,self.model.n)))
        sigma_z = np.matmul(sigma[2].reshape((self.model.nUAVs,1)),np.ones((1,self.model.n)))
        i_picked = 0
        self.inin = 0
        # Check and get rid of no-violation straight-line agents
        RidofAgentNo = []
        RidofBestsolution = []
        position = Position()
        for UAV in range(self.model.nUAVs):
            #straight line from source to destination
            xx = np.linspace(self.model.xs[UAV],self.model.xt[UAV],self.model.n+2)
            yy = np.linspace(self.model.ys[UAV],self.model.yt[UAV],self.model.n+2)
            zz = np.linspace(self.model.zs[UAV],self.model.zt[UAV],self.model.n+2)
            position.x.extend((xx[1:-1]).tolist())
            position.y.extend((yy[1:-1]).tolist())
            position.z.extend((zz[1:-1]).tolist())

        [cost, pathLength, sol] = self.MyCost(position,self.model,elminate=1) # eliminate=1 to output the each UAV's data
        for UAV in range(self.model.nUAVs):    
            if sol.IsFeasible[UAV]:
                # then this agent are following a straight line, so it can be get rid of from pso
                RidofAgentNo.extend(UAV)
                xs = np.delete(xs,UAV)
                xt = np.delete(xt,UAV)
                ys = np.delete(ys,UAV)
                yt = np.delete(yt,UAV)
                zs = np.delete(zs,UAV)
                zt = np.delete(zt,UAV)
                xobs = np.append(xobs,sol.xs[UAV])
                yobs = np.append(yobs,sol.ys[UAV])
                zobs = np.append(zobs,sol.zs[UAV])
                robs = np.append(robs,Config.droneSideLength/2)
                hobs = np.append(hobs,Config.droneSideLength)
                RidofBestsolution.extend(sol.xx[])
                [xx[1*n+1+j*Config.Seperate]
                xx = GlobalBest.Sol.xx
                
        self.model.update_param(xobs, yobs, zobs, robs, hobs, nObs, n, xmin, xmax, ymin, ymax, zmin, zmax, obstBuffer, xs, ys, zs, xt, yt, zt)
            


        # PSO Initialization
        for i in range(nPop):
            self.particle[i] = empty_particle()
            if i > 0:
                self.particle[i].Position = self.CreateRandomSolution(self.model,self.particle[0].Position, sigma_x, sigma_y,sigma_z)
            
            else:
                for j in range(self.model.nUAVs):
                    #straight line from source to destination
                    xx = np.linspace(self.model.xs[j],self.model.xt[j],self.model.n+2)
                    yy = np.linspace(self.model.ys[j],self.model.yt[j],self.model.n+2)
                    zz = np.linspace(self.model.zs[j],self.model.zt[j],self.model.n+2)
                    self.particle[i].Position.x.extend((xx[1:-1]).tolist())
                    self.particle[i].Position.y.extend((yy[1:-1]).tolist())
                    self.particle[i].Position.z.extend((zz[1:-1]).tolist())
            
            self.particle[i].Velocity.x = np.zeros((1,VarSize[1]*self.model.nUAVs))[0]+0.5 #[[0.,0.,0.]][0] = [0,0,0]
            self.particle[i].Velocity.y = np.zeros((1,VarSize[1]*self.model.nUAVs))[0]+0.5
            self.particle[i].Velocity.z = np.zeros((1,VarSize[1]*self.model.nUAVs))[0]+0.5

            # Evaluation
            # print(f'the {i}th particle initilization')
            [self.particle[i].Cost, self.particle[i].PathLength, self.particle[i].Sol] = self.MyCost(self.particle[i].Position,self.model,elminate=0)
            # print(f'Now drawing the {i}th particle')
            # self.PlotSolution(self.particle[i].Sol, self.model,self.MaxIt-1)
            
            self.particle[i].Best.Position.x = self.particle[i].Position.x.copy()
            self.particle[i].Best.Position.y = self.particle[i].Position.y.copy()
            self.particle[i].Best.Position.z = self.particle[i].Position.z.copy()

            self.particle[i].Best.Cost = self.particle[i].Cost.copy()
            self.particle[i].Best.Sol =  copy.deepcopy(self.particle[i].Sol)

            self.particle[i].Best.PathLength = self.particle[i].PathLength.copy()

            # Update Global Best
            if self.particle[i].Best.Cost < self.GlobalBest.Best.Cost:
                self.GlobalBest.Best = copy.deepcopy(self.particle[i].Best)
                # print(f'which particles are the optimal one: {i} initilization')
                # print(f'This best particles Violation is {self.particle[i].Best.Cost-self.particle[i].Best.PathLength} compared with {self.particle[i].Best.Sol.Violation}')
                i_picked = i
        # print(f'GlobalBest after initialization has violation of {self.GlobalBest.Best.Cost-self.GlobalBest.Best.PathLength}')
        # self.PlotSolution(self.GlobalBest.Best.Sol, self.model,self.MaxIt-1)
        # self.PlotSolution(self.particle[18].Best.Sol, self.model,self.MaxIt-1)

        # Array to hold best cost values at each iteration
        BestCost = np.zeros((self.MaxIt,1))
        BestPathLength = np.zeros((self.MaxIt,1))

        # PSO Main Loop
        self.inin = 1
        # model_update = self.model
        break_flg = 0
        for it in range(self.MaxIt):
            if break_flg:
                break
            for i in range(nPop):
                break_flg = 0
                # x part
                # update velocity
                self.particle[i].Velocity.x = w*np.array(self.particle[i].Velocity.x) + \
                    c1*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.particle[i].Best.Position.x)-np.array(self.particle[i].Position.x)))+ \
                                                c2*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.GlobalBest.Best.Position.x)-np.array(self.particle[i].Position.x)))

                # Update velocity bounds
                self.particle[i].Velocity.x = np.maximum(self.particle[i].Velocity.x, self.VelMin.x)
                self.particle[i].Velocity.x = np.minimum(self.particle[i].Velocity.x, self.VelMax.x)

                # Update Position
                self.particle[i].Position.x = self.particle[i].Position.x + self.particle[i].Velocity.x
                # Velocity Mirroring
                OutofTheRange = (np.less(self.particle[i].Position.x, self.VarMin.x).all() and np.greater(self.particle[i].Position.x, self.VarMax.x).all())
                if OutofTheRange == False:
                    self.particle[i].Velocity.x = -self.particle[i].Velocity.x

                # Update Position Bounds
                self.particle[i].Position.x = np.maximum(self.particle[i].Position.x, self.VarMin.x)
                self.particle[i].Position.x = np.minimum(self.particle[i].Position.x, self.VarMax.x)

                # y part
                 # update velocity
                self.particle[i].Velocity.y = w*self.particle[i].Velocity.y + \
                    c1*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.particle[i].Best.Position.y)-np.array(self.particle[i].Position.y)))+ \
                        c2*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.GlobalBest.Best.Position.y)-np.array(self.particle[i].Position.y)))

                # Update velocity bounds
                self.particle[i].Velocity.y = np.maximum(self.particle[i].Velocity.y, self.VelMin.y)
                self.particle[i].Velocity.y = np.minimum(self.particle[i].Velocity.y, self.VelMax.y)

                # Update Position
                self.particle[i].Position.y = self.particle[i].Position.y + self.particle[i].Velocity.y

                # Velocity Mirroring
                OutofTheRange = (np.less(self.particle[i].Position.y, self.VarMin.y).all() and np.greater(self.particle[i].Position.y, self.VarMax.y).all())

                if OutofTheRange == False:
                    self.particle[i].Velocity.y = -self.particle[i].Velocity.y
                
                # Update Position Bounds
                self.particle[i].Position.y = np.maximum(self.particle[i].Position.y, self.VarMin.y)
                self.particle[i].Position.y = np.minimum(self.particle[i].Position.y, self.VarMax.y)    

                # z Part
                # update velocity
                self.particle[i].Velocity.z = w*self.particle[i].Velocity.z + \
                    c1*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.particle[i].Best.Position.z)-np.array(self.particle[i].Position.z)))+ \
                        c2*np.multiply(np.random.rand(1,VarSize[1]*self.model.nUAVs)[0],(np.array(self.GlobalBest.Best.Position.z)-np.array(self.particle[i].Position.z)))

                # Update velocity bounds
                self.particle[i].Velocity.z = np.maximum(self.particle[i].Velocity.z, self.VelMin.z)
                self.particle[i].Velocity.z = np.minimum(self.particle[i].Velocity.z, self.VelMax.z)
                
                # Update Position
                self.particle[i].Position.z = self.particle[i].Position.z + self.particle[i].Velocity.z           
                
                # Velocity Mirroring
                OutofTheRange = (np.less(self.particle[i].Position.z, self.VarMin.z).all() and np.greater(self.particle[i].Position.z, self.VarMin.z).all())
                if OutofTheRange == False:
                    self.particle[i].Velocity.z = -self.particle[i].Velocity.z
                
                # Update Position Bounds
                self.particle[i].Position.z = np.maximum(self.particle[i].Position.z, self.VarMin.z)
                self.particle[i].Position.z = np.minimum(self.particle[i].Position.z, self.VarMax.z) 

                # covert position and velocity to list
                self.temp_particle.Position.x = self.particle[i].Position.x.tolist().copy()
                self.temp_particle.Position.y = self.particle[i].Position.y.tolist().copy()
                self.temp_particle.Position.z = self.particle[i].Position.z.tolist().copy()
                # Evaluation
                [self.particle[i].Cost, self.particle[i].PathLength, self.particle[i].Sol] = self.MyCost(self.temp_particle.Position, self.model,elminate=0)
                # print(f'{self.inin} next particles {i}')
                # Update Personal Best
                if (self.particle[i].Cost <self.particle[i].Best.Cost):
                    self.particle[i].Best = copy.deepcopy(self.particle[i])
                    # Update Global Best
                    if self.particle[i].Best.Cost < self.GlobalBest.Best.Cost:
                        self.GlobalBest.Best = copy.deepcopy(self.particle[i].Best)
            
                        # print(f'WHICH particles have the smallest total cost: {i} ')
                        i_picked = i
                # print(f'Compared {i_picked} with {i}th particles: Violation cost is {self.particle[i].Cost-self.particle[i].PathLength} and lenght is {self.particle[i].PathLength}')
            # print(f'GlobalBest one: the {i_picked}th particles find the optimal trajectory, Violation cost is {self.particle[i_picked].Cost-self.particle[i_picked].PathLength} and lenght is {self.particle[i_picked].PathLength}')
            # print(f'the best solution is at iteration of {it}th at {i_picked}th particle, Violation is {self.GlobalBest.Best.Sol.Violation} and feasible is {self.GlobalBest.Best.Sol.IsFeasible}')
           
            # Update Best Cost Ever Found
            BestCost[it] = self.GlobalBest.Best.Cost.copy()
            
            BestPathLength[it] = self.GlobalBest.Best.PathLength.copy()

            print(f'Violation is {BestCost[it]-BestPathLength[it]}')

            # Inertia Weight Damping
            w = w*wdamp

            # Show Iteration Information
            if self.GlobalBest.Best.Sol.IsFeasible:
                Flag = '*'
            else:
                Flag = (",Violation = " + str(self.GlobalBest.Best.Sol.Violation))
            # print(f'Iteration {it}: Best Cost = {BestCost[it]} {Flag}')
            # self.PlotSolution(self.GlobalBest.Best.Sol, self.model,self.MaxIt-1)
            if self.GlobalBest.Best.Sol.IsFeasible:
                break_flg = 1
                break
        if it > 0:
            print(f'not a straight line after iteration of {it}')
        # return self.GlobalBest.Best, model_update
        return self.GlobalBest.Best

    def CreateCylinder(self,single_xobs,single_yobs,single_zobs,single_hobs,single_robs):
        
        # axis and radius
        p1 = np.array([single_xobs,single_yobs,single_zobs])
        # vector in direction of axis
        v = np.array([0,0,single_hobs])
        p0 = p1-v
        R = single_robs
        # find magnitude of vector
        mag = norm(v)
        # unit vector in direction of axis
        v = v/mag
        #make some vector not in the same direction as v
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
        self.bx.plot_surface(X, Y, Z)
        #plot axis
        self.bx.plot(*zip(p0, p1), color = 'red')


    def PlotSolution(self, sol, model, iteraNo):
        fig, self.bx = plt.subplots(subplot_kw={"projection": "3d"})
        
        xs = model.xs
        ys = model.ys
        zs = model.zs
        xt = model.xt
        yt = model.yt
        zt = model.zt
        xobs = model.xobs
        yobs = model.yobs
        zobs = model.zobs
        hobs = model.hobs
        # robs = [x - model.obstBuffer for x in model.robs]
        robs = model.robs
        nVar = model.n #if wanna plot determined points

        XS = sol.XS
        YS = sol.YS
        ZS = sol.ZS
        xx = sol.xx
        yy = sol.yy
        zz = sol.zz

        for k in prange(len(xobs)):
            self.CreateCylinder(xobs[k],yobs[k],zobs[k],hobs[k],robs[k])
        nUAVs = self.model.nUAVs
        
        self.bx.set_xlabel('X')
        self.bx.set_ylabel('Y')
        self.bx.set_zlabel('Z')

        nc = matplotlib.colors.Normalize(vmin=0, vmax=nUAVs)
        for i in prange(nUAVs):
            self.bx.plot3D(xx[(Config.Seperate*i):(Config.Seperate*(i+1))], yy[(Config.Seperate*i):(Config.Seperate*(i+1))], zz[(Config.Seperate*i):(Config.Seperate*(i+1))],marker='x') 
            # self.bx.plot3D(XS[((nVar+2)*i):((nVar+2)*(i+1))], YS[((nVar+2)*i):((nVar+2)*(i+1))], ZS[((nVar+2)*i):((nVar+2)*(i+1))],marker='x') #if wanna plot determined points

        self.bx.scatter(xs,ys,zs,marker='*')
        self.bx.scatter(xt,yt,zt,marker='o')
        
        if (iteraNo == (self.MaxIt-1)):
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()


    def ParseSolution(self, sol1, model, elminate):

        nUAVs = model.nUAVs
        nVar = model.n
        Violation = 0
        Violation_each = np.zeros((nUAVs,1))
        XS = []
        YS = []
        ZS = []
        L = 0
        L_each = 0
        temp_xx = []
        temp_yy = []
        temp_zz = []

        xobs = [] if model.xobs is None else model.xobs.tolist().copy()
        yobs = [] if model.yobs is None else model.yobs.tolist().copy()
        zobs = [] if model.zobs is None else model.zobs.tolist().copy()
        robs = [] if model.robs is None else model.robs.tolist().copy()
        hobs = [] if model.hobs is None else model.hobs.tolist().copy()

        x = np.array(sol1.x).reshape((nUAVs,nVar))
        x_temp = np.hstack((model.xs.reshape((nUAVs,1)), x)) 
        x_temp = np.hstack((x_temp, model.xt.reshape((nUAVs,1)))) #[xs, x..., xt]*nUAVs
        XS = np.hstack((x_temp)) #if wanna plot determined points

        y = np.array(sol1.y).reshape((nUAVs,nVar))
        y_temp = np.hstack((model.ys.reshape((nUAVs,1)), y)) 
        y_temp = np.hstack((y_temp, model.yt.reshape((nUAVs,1)))) #[ys, y..., yt]*nUAVs
        YS = np.hstack((y_temp)) #if wanna plot determined points

        z = np.array(sol1.z).reshape((nUAVs,nVar))
        z_temp = np.hstack((model.zs.reshape((nUAVs,1)), z)) 
        z_temp = np.hstack((z_temp, model.zt.reshape((nUAVs,1)))) #[zs, z..., zt]*nUAVs
        ZS = np.hstack((z_temp)) #if wanna plot determined points

        Pos_XYZ = np.stack((x_temp, y_temp, z_temp))
        k = nVar+2
        TS = np.linspace(0,1,k)
        # tt = np.linspace(0,1,100)
        tt = np.linspace(0,1,Config.Seperate)
        n = len(tt) # number of points to be seperated
        spl = interp1d(TS, Pos_XYZ[:], kind='cubic')
        xxyyzz = spl(tt)
        dxyz = np.diff(xxyyzz)
        X = dxyz[0,:]
        Y = dxyz[1,:]
        Z = dxyz[2,:]

        temp_xx = np.hstack(xxyyzz[0])
        temp_yy = np.hstack(xxyyzz[1])
        temp_zz = np.hstack(xxyyzz[2])

        L = np.sum(np.sqrt(X**2+Y**2+Z**2))
        L_each = np.sum(np.sqrt(X**2+Y**2+Z**2), axis=1)
        if len(L_each) == model.nUAVs:
            print(f'path length has right computation')

        for UAV in range(nUAVs):
            xx = xxyyzz[0,UAV]
            yy = xxyyzz[1,UAV]
            zz = xxyyzz[2,UAV]
            nobs = len(xobs) # number of obstacles
            
            for k in range(nobs):
                xx_filtered = []
                yy_filtered = []

                for j in range(n):
                    if(zz[j] <= zobs[k]) and (zz[j] >= zobs[k]-hobs[k]):
                        xx_filtered.append(xx[j])
                        yy_filtered.append(yy[j])
                # print(f'xx_filtered is {xx_filtered} and yy_filtered is {yy_filtered} and xobs are {xobs} and yobs are {yobs}')
                d = ((np.array(xx_filtered)-np.array(xobs)[k])**2 + (np.array(yy_filtered)-yobs[k])**2)**0.5
                temp = []
                if (robs[k] != 0):
                    temp = 1-d/(robs[k]+model.obstBuffer/2) # By considering sidelength of drone
                zero_array = np.zeros_like(temp)
                v = np.maximum(temp,zero_array)
                if (len(v)!=0):
                    Violation_each[UAV] = Violation[UAV] +np.mean(v)
                    Violation = Violation + np.mean(v)
                    # if Violation != 0:
                        # print(f'obstacle-avoidance is {Violation} of UAV {UAV} and obstacle {k}')
                if (self.inin == 1):
                    print(f'{k}th obstacle radius is {robs[k]} and d is {d} and violation array is {v}')
                    print(f'obstacle-avoidance is {Violation_each[UAV]} of UAV {UAV} and obstacle {k}')
                if(math.isnan(Violation)):
                    print("STOP")
            if (elminate!=1):
                # xobs.extend(xx.tolist()[9:(Config.Seperate-10)]) #Seperate was 100 
                xobs.extend(xx.tolist()[2:(Config.Seperate-2)])
                # yobs.extend(yy.tolist()[9:(Config.Seperate-10)])
                yobs.extend(yy.tolist()[2:(Config.Seperate-2)])
                # zobs.extend((zz[9:90]+(self.model.obstBuffer+0.15)).tolist()) # agent viewed as cylinderical shape
                # zobs.extend((zz[9:(Config.Seperate-10)]).tolist())
                zobs.extend((zz[2:(Config.Seperate-2)]).tolist())
                # robs.extend((self.model.obstBuffer+0.15)*np.ones(81)) # agent viewed as cylinderical shape
                robs.extend(Config.droneSideLength/2*np.ones(Config.Seperate-2-2)) #agent side length = Config.droneSideLength
                # robs.extend(0.15/2*np.ones(Config.Seperate-10-9)) #agent side length = 0.15m
                # hobs.extend((self.model.obstBuffer+0.15)*2*np.ones(81)) # agent viewed as cylinderical shape
                # hobs.extend(0.15*np.ones(Config.Seperate-10-9))
                hobs.extend(Config.droneSideLength*np.ones(Config.Seperate-2-2))

        self.sol2.update_param(TS, XS, YS, ZS, tt, temp_xx, temp_yy, temp_zz, L, L_each, Violation, Violation_each)
        sol = self.sol2

        return sol

    def CreateRandomSolution(self, model, position, sigma_x,sigma_y,sigma_z):
        pos_x_ref = np.array(position.x).reshape((model.nUAVs,model.n))
        pos_y_ref = np.array(position.y).reshape((model.nUAVs,model.n))
        pos_z_ref = np.array(position.z).reshape((model.nUAVs,model.n))

        x_temp = np.random.uniform(pos_x_ref-sigma_x, pos_x_ref+sigma_x)# [x0,x1,x2..model.n]*nUAVs
        y_temp = np.random.uniform(pos_y_ref-sigma_y, pos_y_ref+sigma_y)
        z_temp = np.random.uniform(pos_z_ref-sigma_z, pos_z_ref+sigma_z)

        sol = Position()
        sol.x, sol.y, sol.z = np.hstack(x_temp), np.hstack(y_temp), np.hstack(z_temp)
        return sol


    def MyCost(self, sol1, model, elminate):
        sol = self.ParseSolution(sol1,model, elminate) #class sol2()
        # beta = 10
        if (elminate==1):
            z = sol.L_each*(1+Config.beta*sol.Violation_each)
            zl = sol.L_each
        else:
            z = sol.L*(1+Config.beta*sol.Violation)
            zl = sol.L

        return [z, zl, sol]

if __name__ == '__main__':
    starttime = timeit.default_timer()
    print("The start time is :",starttime)  
    np.random.seed(1)
    random.seed(1)
    Path_Generation = path_generation()
    
    # xobs = np.array([-1.8, -1, -0.5,    0,      0.5,    1.7,  0])*5
    # yobs = np.array([-1.8,  0,  1.5,    -0.5,   1.5,    -0.5, -2])*5
    # zobs = np.array([2,   1.8,    1,    0.5,    1.5,    1.3,  2])*5
    # robs = np.array([0.3, 0.1,  0.5,    0.2,    0.4,    0.5,  0.5])*5
    # # hobs = np.array([1,   0.2,  0.6,    0.8,    0.9,       2,  2])
    # hobs = np.array([2,   2,  2,   2,    2,    2,  2])*5
    xobs = np.array([-1.8])
    yobs = np.array([-1.8])
    zobs = np.array([10])
    robs = np.array([1])
    hobs = np.array([10])

    nObs = len(xobs)
    xmin = -6
    # xmin = -2
    xmax = 6
    # xmax = 2
    ymin = -6
    # ymin = -2
    ymax = 6
    # ymax = 2
    zmin = 0
    zmax = 6
    # zmax = 2
    # xs = np.array([1.5, 1.5, 1.5, 1.5])
    # ys = np.array([1.5, 0.5, -0.5, -1.5])
    # zs = np.array([0, 0, 0, 0])
    # xs = np.array([-1.5, -2, -1.5])*5
    xs = np.array([-3])
    # xs = np.array([-1.5, -2, -1.5])
    # ys = np.array([-1.7, -1.5, -1.5])
    # ys = np.array([-1.7, -1.5, -1.5])*5
    ys = np.array([-3])
    # zs = np.array([0, 0, 0])
    # zs = np.array([0, 0, 0])*5
    zs = np.array([0])
    # target_init = np.array([1.5, 0.0, 1.6])
    # xt = np.array([1.3, 1.5, 1.5, 1.0]) 
    # yt = np.array([0.0, 1.0, 0.0, 0.0]) 
    # zt = np.array([1.6, 1.6, 1.8, 2.0])  
    # xt = np.array([1.9, 1.7, 1.3])*5
    xt = np.array([2])
    # xt = np.array([1.9, 1.7, 1.6])
    # yt = np.array([1.5, 1.8, 1.1])*5
    yt = np.array([2])
    # yt = np.array([1.5, 1.8, 1.4])
    zt = np.array([2])
    # zt = np.array([1.8, 1.5, 1.4])*5
    # zt = np.array([1.8, 1.5, 1.2])
    starttime_pso = time.time()
    Path_Generation.pso(xobs, yobs, zobs, robs, hobs, nObs, xmin, xmax, ymin, ymax, zmin, zmax, xs, ys, zs, xt, yt, zt)
    print(f'pso spent {time.time() - starttime_pso:6.4}')   









