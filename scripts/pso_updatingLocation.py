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
        self.PathLength = []
        self.Best = Best()
        self.Position = Position()
        self.Sol = sol2()

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

    def update_param(self, TS, XS, YS, ZS, tt, xx, yy, zz, L, Violation):
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
        self.Violation = Violation
        self.IsFeasible = (self.Violation==0)

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
        obstBuffer = droneSideLength*0

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
        self.VelMax.x = alpha*(self.VarMax.x-self.VarMin.x) # Maximum Velocity
        self.VelMin.x = -self.VelMax.x                # Minimum Velocity
        self.VelMax.y =  alpha*(self.VarMax.y-self.VarMin.y) # Maximum Velocity
        self.VelMin.y = -self.VelMax.y                # Minimum Velocity
        self.VelMax.z =  alpha*(self.VarMax.z-self.VarMin.z) # Maximum Velocity
        self.VelMin.z = -self.VelMax.z                # Minimum Velocity        
        
        # Initialize Global Best
        self.GlobalBest.Best.Cost = np.Inf

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
            [self.particle[i].Cost, self.particle[i].PathLength, self.particle[i].Sol] = self.MyCost(self.particle[i].Position,self.model)
            
            self.particle[i].Best.Position.x = self.particle[i].Position.x.copy()
            self.particle[i].Best.Position.y = self.particle[i].Position.y.copy()
            self.particle[i].Best.Position.z = self.particle[i].Position.z.copy()

            self.particle[i].Best.Cost = self.particle[i].Cost.copy()
            self.particle[i].Best.Sol =  copy.deepcopy(self.particle[i].Sol)

            self.particle[i].Best.PathLength = self.particle[i].PathLength.copy()

            # Update Global Best
            if self.particle[i].Best.Cost < self.GlobalBest.Best.Cost:
                self.GlobalBest.Best = copy.deepcopy(self.particle[i].Best)

        # Array to hold best cost values at each iteration
        BestCost = np.zeros((self.MaxIt,1))
        BestPathLength = np.zeros((self.MaxIt,1))

        # PSO Main Loop
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
                [self.particle[i].Cost, self.particle[i].PathLength, self.particle[i].Sol] = self.MyCost(self.temp_particle.Position, self.model)
                
                # Update Personal Best
                if (self.particle[i].Cost <self.particle[i].Best.Cost):
                    self.particle[i].Best = copy.deepcopy(self.particle[i])
                    # Update Global Best
                    if self.particle[i].Best.Cost < self.GlobalBest.Best.Cost:
                        self.GlobalBest.Best = copy.deepcopy(self.particle[i].Best)

            # Update Best Cost Ever Found
            BestCost[it] = self.GlobalBest.Best.Cost.copy()
            
            BestPathLength[it] = self.GlobalBest.Best.PathLength.copy()

            # Inertia Weight Damping
            w = w*wdamp

            # Show Iteration Information
            if self.GlobalBest.Best.Sol.IsFeasible:
                Flag = '*'
            else:
                Flag = (",Violation = " + str(self.GlobalBest.Best.Sol.Violation))
            print(f'Iteration {it}: Best Cost = {BestCost[it]} {Flag}')

            if self.GlobalBest.Best.Sol.IsFeasible:
                break_flg = 1
                break

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
        robs = [x - model.obstBuffer for x in model.robs]
        nVar = model.n

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
            self.bx.plot3D(XS[((nVar+2)*i):((nVar+2)*(i+1))], YS[((nVar+2)*i):((nVar+2)*(i+1))], ZS[((nVar+2)*i):((nVar+2)*(i+1))],marker='x') 

        self.bx.scatter(xs,ys,zs,marker='*')
        self.bx.scatter(xt,yt,zt,marker='o')
        
        if (iteraNo == (self.MaxIt-1)):
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()


    def ParseSolution(self, sol1, model):

        nUAVs = model.nUAVs
        nVar = model.n
        Violation = 0
        XS = []
        YS = []
        ZS = []
        L = 0
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
        XS = np.hstack((x_temp))

        y = np.array(sol1.y).reshape((nUAVs,nVar))
        y_temp = np.hstack((model.ys.reshape((nUAVs,1)), y)) 
        y_temp = np.hstack((y_temp, model.yt.reshape((nUAVs,1)))) #[ys, y..., yt]*nUAVs
        YS = np.hstack((y_temp))
        z = np.array(sol1.z).reshape((nUAVs,nVar))
        z_temp = np.hstack((model.zs.reshape((nUAVs,1)), z)) 
        z_temp = np.hstack((z_temp, model.zt.reshape((nUAVs,1)))) #[zs, z..., zt]*nUAVs
        ZS = np.hstack((z_temp))

        Pos_XYZ = np.stack((x_temp, y_temp, z_temp))
        k = nVar+2
        TS = np.linspace(0,1,k)
        tt = np.linspace(0,1,100)
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
                d = ((np.array(xx_filtered)-np.array(xobs)[k])**2 + (np.array(yy_filtered)-yobs[k])**2)**0.5
                temp = []
                if (robs[k] != 0):
                    temp = 1-d/robs[k]
                zero_array = np.zeros_like(temp)
                v = np.maximum(temp,zero_array)
                if (len(v)!=0):
                    Violation = Violation + np.mean(v)
                if(math.isnan(Violation)):
                    print("STOP")
                
            xobs.extend(xx.tolist()[9:90])
            yobs.extend(yy.tolist()[9:90])
            # zobs.extend((zz[9:90]+(self.model.obstBuffer+0.15)).tolist()) # agent viewed as cylinderical shape
            zobs.extend((zz[9:90]).tolist())
            # robs.extend((self.model.obstBuffer+0.15)*np.ones(81)) # agent viewed as cylinderical shape
            robs.extend(0.15/2*np.ones(81)) #agent side length = 0.15m
            # hobs.extend((self.model.obstBuffer+0.15)*2*np.ones(81)) # agent viewed as cylinderical shape
            hobs.extend(0.15*np.ones(81))


        self.sol2.update_param(TS, XS, YS, ZS, tt, temp_xx, temp_yy, temp_zz, L, Violation)
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


    def MyCost(self, sol1, model):
        sol = self.ParseSolution(sol1,model) #class sol2()
        beta = 10
        z = sol.L*(1+beta*sol.Violation)
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
    xobs = np.array([-2])
    yobs = np.array([-2])
    zobs = np.array([0.1])
    robs = np.array([0.01])
    hobs = np.array([0.01])

    nObs = len(xobs)
    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2
    zmin = 0
    zmax = 2
    xs = np.array([1.5, 1.5, 1.5, 1.5])
    ys = np.array([1.5, 0.5, -0.5, -1.5])
    zs = np.array([0, 0, 0, 0])
    # xs = np.array([-1.5, -2, -1.5])*5
    # xs = np.array([-1.5])*5
    # xs = np.array([-1.5, -2, -1.5])
    # ys = np.array([-1.7, -1.5, -1.5])
    # ys = np.array([-1.7, -1.5, -1.5])*5
    # ys = np.array([-1.7])*5
    # zs = np.array([0, 0, 0])
    # zs = np.array([0, 0, 0])*5
    # zs = np.array([0])*5
    # target_init = np.array([1.5, 0.0, 1.6])
    xt = np.array([1.3, 1.5, 1.5, 1.0]) 
    yt = np.array([0.0, 1.0, 0.0, 0.0]) 
    zt = np.array([1.6, 1.6, 1.8, 2.0])  
    # xt = np.array([1.9, 1.7, 1.3])*5
    # xt = np.array([1.9])*5
    # xt = np.array([1.9, 1.7, 1.6])
    # yt = np.array([1.5, 1.8, 1.1])*5
    # yt = np.array([1.5])*5
    # yt = np.array([1.5, 1.8, 1.4])
    # zt = np.array([1.8])*5
    # zt = np.array([1.8, 1.5, 1.4])*5
    # zt = np.array([1.8, 1.5, 1.2])
    starttime_pso = time.time()
    Path_Generation.pso(xobs, yobs, zobs, robs, hobs, nObs, xmin, xmax, ymin, ymax, zmin, zmax, xs, ys, zs, xt, yt, zt)
    print(f'pso spent {time.time() - starttime_pso:6.4}')   









