import numpy as np
import numpy.linalg as la
# from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from Config import Config

'''
Adjusted from Jacky's 2D path planner into 3D 
algorithm:
    Initialize the planner given map dimensions and time horizon
    along with their discretization stages
    
    take the time, obstacle and target states and start populating the cost
    
    at each timestep. check if the obstacle and target states are consistent with the 
    estimates, if not replan
    
    when filling in the agents, use the decentralized A* search with a
    reservation table. Or rather, update the cost grid with the agent path,
    and to plan for the next agent, do min pooling at the agent location.

    (turns out the last part doesn't affect the solution by any significant amount, removed to improve runtime)
'''

def interpolate(points, lookahead):
    '''
    Parameters
    ----------
    points : 2D array of points, each row is [t,x,y]

    Returns: desired state [x,y,vx,vy,ax,ay]
    -------
    
    '''
    x = IUS(points[:,0], points[:,1])
    y = IUS(points[:,0], points[:,2])
    x.set_smoothing_factor(0.5)
    y.set_smoothing_factor(0.5)
    vx = x.derivative()
    vy = y.derivative()
    ax = vx.derivative()
    ay = vy.derivative()
    t = points[0,0] + lookahead
    return np.array([x(t),y(t),vx(t),vy(t),ax(t),ay(t)])


def min_pool(grid):
    '''
    Perform min pooling with inf padding
    '''
    print('min_pool....')
    # convert to float if int
    if grid.dtype.kind == 'i':
        grid = grid.astype('float')
    
    grid = np.pad(grid, pad_width=1, mode='constant', constant_values=np.inf)
    min_grid = [grid[1:-1,1:-1,1:-1]] + [grid[1:-1,1:-1,2:]] + [grid[1:-1,1:-1,:-2]] + \
                [grid[2:,1:-1,1:-1]] + [grid[:-2,1:-1,1:-1]] + [grid[1:-1,:-2,1:-1]] + [grid[1:-1,2:,1:-1]] 
                
    # 8 connectivity
    # min_grid += [grid[0:-2,0:-2]] + [grid[0:-2,2:]] + [grid[2:,0:-2]] + [grid[2:,2:]]
    
    min_grid = np.min(np.array(min_grid), axis=0)
    return min_grid

def propagate(grid):
    '''
    Run wavefront propagation on the input grid
    
    Parameters
    ----------
    grid : 3D array
        An occupancy grid with the goal at 0, occupied cells at -1, and
        unexplored cells at inf

    Returns
    -------
    filled_grid : 3D array
        populated 3D array
    '''
    # pad the grid with -1s to encode boundaries, remove padding at the end
    grid = np.pad(grid, pad_width=1, mode='constant', constant_values=-1)
    
    # initialize open set to goal, closed set to obstacles
    open_set = set(map(tuple, np.array(np.where(grid == 0)).T))
    closed_set = set(map(tuple, np.array(np.where(grid == -1)).T))
    cost = 0
    open_set = open_set - closed_set

    print('initial open set {}'.format(open_set))
    
    while len(open_set) != 0:
        print('current open set {}\n'.format(open_set))
        print('NEW\n')
        # convert the open set into an array and assign
        open_array = np.array(list(open_set))
        grid[open_array[:,0], open_array[:,1], open_array[:,2]] = cost
        cost += 1
        # add the open set to the closed set
        closed_set = closed_set | open_set
        
        # get the neighbouring set
        neighbour_set = set(map(tuple,open_array + [0,0,-1])) 
        neighbour_set = neighbour_set | set(map(tuple,open_array + [0,0,1]))
        neighbour_set = neighbour_set | set(map(tuple,open_array + [0,1,0]))
        neighbour_set = neighbour_set | set(map(tuple,open_array + [0,-1,0]))
        neighbour_set = neighbour_set | set(map(tuple,open_array + [-1,0,0]))
        neighbour_set = neighbour_set | set(map(tuple,open_array + [1,0,0]))
        # neighbour_set = neighbour_set | set(map(tuple,open_array + [-1,0]))
        # neighbour_set = neighbour_set | set(map(tuple,open_array + [1,0]))
        # neighbour_set = neighbour_set | set(map(tuple,open_array + [0,1]))
        
        # 8 connectivity
        # neighbour_set = neighbour_set | set(map(tuple,open_array + [-1,-1]))
        # neighbour_set = neighbour_set | set(map(tuple,open_array + [-1,1]))
        # neighbour_set = neighbour_set | set(map(tuple,open_array + [1,-1]))
        # neighbour_set = neighbour_set | set(map(tuple,open_array + [1,1]))
        
        # update the open set
        neighbour_set = neighbour_set - closed_set
        open_set = neighbour_set
    
    return grid[1:-1,1:-1,1:-1]
    
class Planner():
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, dt, horizon, window, 
                 j_t, j_s, obs_padding, d_replan, v_replan, t_replan):
        '''
        Initialize the planner given map dimensions, time horizon, and their
        discretization sizes, check the yaml file for parameter descriptions
        
        assume that the map limits fit perfectly within bounds
        '''
        self.x_min, self.x_max, self.dx = x_min, x_max, dx
        self.y_min, self.y_max, self.dy = y_min, y_max, dy
        self.z_min, self.z_max, self.dz = z_min, z_max, dz
        self.dt = dt
        self.horizon = horizon
        self.window = window
        self.j_t, self.j_s = j_t, j_s
        self.obs_padding = obs_padding
        self.d_replan, self.v_replan, self.t_replan = d_replan, v_replan, t_replan

        # cost grid dimensions
        self.Nt = int(self.horizon/self.dt) + 1
        self.Nx = int((self.x_max - self.x_min)/self.dx) + 1
        self.Ny = int((self.y_max - self.y_min)/self.dy) + 1
        self.Nz = int((self.z_max - self.z_min)/self.dz) + 1

        # initialize 4d cost grid with np.inf
        # at each timestep, x axis points down, y axis points right, z axis points outward
        self.cost = np.ones((self.Nt, self.Nx, self.Ny, self.Nz))*np.inf

        # parameters for the planner
        self.t_min, self.t_max = -1, -1
        self.origin = np.array([self.t_min, self.x_min, self.y_min, self.z_min])
        self.target_init = np.array([0,0,0,0,0,0]) #[x,y,z,vx,vy,vz]
        # self.target_init = np.array([0,0,0,0])
        self.obstacle_init = None 
        
        # stored data after planning
        # 4D array where each index is a 3D array with obstacle states at t
        # each row in each 3D array is [t,x,y,vx,vy,r] [t,x,y,z,h,r,vx,vy]
        self.obstacle_states = None
        
        # 2D array of target states, each row is [t,x,y,vx,vy] [t,x,y,z,vx,vy,vz]
        self.target_states = None
        
    def plan(self, t, order, agent_state, target_state, obstacle_state):
        '''
        Parameters
        ----------
        t: time
        agent_state: 2D array of agent states, each row is [x,y,vx,vy]           [x,y,z,vx,vy,vz]
        target_state: 2D array of the target state [[x,y,vx,vy]]                [[x,y,z,vx,vy,vz]]
        obstacle_state: 2D array of obstacle states, each row is [x,y,vx,vy,r]   [x,y,z,h,r,vx,vy]

        Returns
        -------
        A set of waypoints for each agent in the form of a 3D array
        each index contains a 2d array of waypoints, each row is [t,x,y,z]
        '''
        replan = False
        # check whether or not we need to replan, by comparing the time
        # and the actual states with the predicted states
        if self.t_max < (t + self.window):
            replan = True
        else:
            # check if we need to replan due to obstacle estimation error
            if obstacle_state is not None:
                vel_diff = la.norm(self.obstacle_init[:,5:7]-obstacle_state[:,5:7], axis=1)
                if np.any(vel_diff > self.v_replan):
                    replan = True
                    print('replan due to vel diff')
                else:
                    pred_obs = self.obstacle_init[:,0:2] + self.obstacle_init[:,5:7]*(t-self.t_min)
                    dist_diff = la.norm(pred_obs-obstacle_state[:,0:2], axis=1)
                    if np.any(dist_diff > self.d_replan):
                        replan = True
                        print('replan due to dist diff')

            # check if we need to replan due to target estimation error
            if not replan:
                vel_diff = np.linalg.norm(self.target_init[0,3:6]-target_state[0,3:6])
                if vel_diff > self.v_replan:
                    replan = True
                else:
                    pred_target = self.target_init[0,0:3] + self.target_init[0,3:6]*(t-self.t_min) # assume target is moving at a horizontal plane
                    dist_diffs = la.norm(pred_target-target_state[0,0:3])
                    if dist_diffs > self.d_replan:
                        replan = True

        # reset the time and initial states, and recompute the cost grid
        if replan:
            print('replanning...')
            self.t_min = np.floor(t/self.dt)*self.dt
            self.t_max = np.floor((t+self.horizon)/self.dt)*self.dt
            self.origin[0] = self.t_min
            self.target_init = np.copy(target_state)
            self.target_init[0,0:3] += self.target_init[0,3:6]*(self.t_min - t)
            if obstacle_state is not None:
                self.obstacle_init = np.copy(obstacle_state)
                self.obstacle_init[:,0:2] += self.obstacle_init[:,5:7]*(self.t_min - t) # x+vx*t, y+vy*t
                self.obstacle_init[:,4] += self.obs_padding #padding obstacle radius by obs_padding, r+padding

            # self.cost = np.ones((self.Nt, self.Nx, self.Ny, self.Nz))*np.inf solve this later Jan10,2022
            print('reach here...')
            self.fill_grid()
            print('Done fill_grid.....')

        return self.search(t, order, agent_state)

    def fill_grid(self):
        '''
        Run value iteration on the 3d grid, using the class parameters
        '''
        # 1) step the target and obstacles through time, and populate its values
        timesteps = np.arange(self.t_min, self.t_max + self.dt, self.dt)

        if timesteps[-1] > (self.t_max + self.dt/2):
            timesteps = timesteps[:-1]

        # a 2D array where each row is [t,x,y,z] indicating the position occupied
        target_states = np.vstack((timesteps, 
                                   self.target_init[0,0] + self.target_init[0,3]*(timesteps - self.t_min),
                                   self.target_init[0,1] + self.target_init[0,4]*(timesteps- self.t_min),
                                   self.target_init[0,2] + self.target_init[0,5]*(timesteps- self.t_min))).T
        # assign target cells, put limits on the target_states
        target_states[:,1] = np.clip(target_states[:,1], self.x_min, self.x_max)
        target_states[:,2] = np.clip(target_states[:,2], self.y_min, self.y_max)
        target_states[:,3] = np.clip(target_states[:,3], self.z_min, self.z_max)

        target_idx = self.coords_to_idx(target_states) 
        self.cost[target_idx[:,0],target_idx[:,1],target_idx[:,2],target_idx[:,3]] = 0

        self.target_states = target_states 

        # a 3D array where each slice is a 2D array [x,y,r] indicating the centers
        # each slice represents the centers at an instance in time

        if self.obstacle_init is not None:
            obstacle_states = np.tile(self.obstacle_init[:,:],(len(timesteps),1,1)) 
            obstacle_states[:,:,0:2] += (timesteps - self.t_min)[:,np.newaxis,np.newaxis]*self.obstacle_init[:,5:7]
            self.obstacle_states = obstacle_states

            #self.obstacle_state = obstacle_states
            # generate grid coordinates for distance comparision       
            zs, ys, xs = np.meshgrid(np.arange(self.z_min, self.z_max+self.dz, self.dz),np.arange(self.y_min, self.y_max+self.dy, self.dy), np.arange(self.x_min, self.x_max+self.dx, self.dx))
            xs, ys, zs = xs.reshape([-1,1]), ys.reshape([-1,1]), zs.reshape([-1,1])

            # I'll vectorize this loop later, it fills in obstacle indices on the cost grid 
            for t,s in enumerate(obstacle_states):
                # get distance squared of every point to the obstacle centers by broadcasting
                l2_squared = (xs - s[:,0])**2 + (ys - s[:,1])**2    # horizontal
                l2_vertical_squared = (zs - s[:,2])**2              # vertical 2D does not consider this parameter, so leave it for now
                
                # check if each point is inside an obstacle, this is a Nx*Ny 1D bool array
                occupancy = np.any(l2_squared < s[:,4]**2, axis=1)
                
                # reshape the occupancy array and assign to cost
                occupancy = occupancy.reshape([self.Nx, self.Ny, self.Nz])
                self.cost[t][occupancy] = -1

                # if the obstacles do not move, just copy the layers
                if not np.any(self.obstacle_init[:,5:7]):
                    for i in range(len(self.cost)):
                        self.cost[i][occupancy] = -1
                    break

        # 2) run wavefront propagation on the last layer
        self.cost[-1] = propagate(self.cost[-1])   
        print('Done propagation....')
        self.cost[-1][self.cost[-1] == -1] = np.inf

        # if no obstacles, or if obstacles and target do not move, just copy the last layer
        if self.obstacle_init is None and (not np.any(self.target_init[0,5:7])):
            self.cost[:] = self.cost[-1]
        else:
            # 3) min pool back in time, reassigning obstacles to inf
            # filling the first layer is unnecessary, but done for consistency
            for t in range(self.Nt-2,-1,-1):
                wait_cost = self.cost[t+1] + self.j_t
                move_cost = min_pool(self.cost[t+1]) + self.j_s
                min_cost = np.minimum(wait_cost, move_cost)
                self.cost[t][self.cost[t] > 0] = min_cost[self.cost[t] > 0] 
                self.cost[t][self.cost[t] == -1] = np.inf
        

    def search(self, t, order, agent_state):
        '''
        Parameters
        ----------
        t: time to begin search
        order: order of priority for search as a 1D array, 1 is top priority
        if any elements are 0, the corresponding agents maintain current velocity
        agent_state: 2D array of agent coordinates where each row is [x,y,z,vx,vy,vz]

        Returns
        -------
        paths: list of 2D arrays
            each element in the list specifies what each agent should do
            if paths[i] = None then agent i is not in the planning phase
            otherwise paths[i] is a 2D array of waypoints, each row is [t,x,y,z]
        '''
        if agent_state.ndim == 1:
            agent_state = agent_state.reshape([1,-1])

        # get the time index of the search
        start_t_idx = int(np.ceil((t-self.t_min)/self.dt))
        end_t_idx = int(start_t_idx + self.window/self.dt)
        end_t_idx = min([end_t_idx, self.Nt-1])


        # Old implementation of local repair search
        coords = np.zeros((len(agent_state), 4)) #[t,x,y,z]
        coords[:,0] = t
        coords[:,1:] = agent_state[:,0:3]

        # path_idx is a 3D array of indices containing the optimal path, i.e. the index of path_coords
        path_idx = np.zeros((end_t_idx-start_t_idx + 1, coords.shape[0], coords.shape[1]), dtype='int')
        path_idx[0] = self.coords_to_idx(coords)
        path_coords = np.zeros_like(path_idx, dtype='float')
        path_coords[0] = np.copy(coords)
        
        # indexing arrays, we don't use [-1,0,1] since we pad the cost
        zs, ys, xs = np.meshgrid([0,1,2], [0,1,2], [0,1,2])
        xs, ys, zs= xs.flatten(), ys.flatten(), zs.flatten()
        
        # search forward in time to fill both path_idx and path_coords
        # rt is the relative time index, used for indexing the path array
        for t in range(start_t_idx + 1, end_t_idx + 1):
            rt = t - start_t_idx
            # pad the next cost to handle boundaries
            padded_cost = np.pad(self.cost[t], pad_width=1, mode='constant', constant_values=np.inf)
            
            # get all cells to search for in the next timestep, reshape path_idx to broadcast
            x_next = (xs + path_idx[rt-1][:,1].reshape([-1,1]))
            y_next = (ys + path_idx[rt-1][:,2].reshape([-1,1]))
            z_next = (zs + path_idx[rt-1][:,3].reshape([-1,1]))

            # get the cost at the next layer, and find the minimum index
            cost_next = padded_cost[x_next.flatten(), y_next.flatten(), z_next.flatten()]
            # cost_next = cost_next.reshape([-1,9])        #correction reach there
            cost_next = cost_next.reshape([1,-1])        #correction reach there
            
            # get the minimum linear index, then unravel to get index differences
            min_lindex = np.argmin(cost_next, axis=1)
            print(min_lindex)
            dx, dy = np.unravel_index(min_lindex,(3,3)) #ValueError: invalid entry in index array
            
            # assign the next path index and coordinates
            path_idx[rt,:,0] = t
            path_idx[rt,:,1] = path_idx[rt-1,:,1] + dx - 1
            path_idx[rt,:,2] = path_idx[rt-1,:,2] + dy - 1
            path_idx[rt,:,3] = path_idx[rt-1,:,3] + dy - 1
            path_coords[rt] = self.idx_to_coords(path_idx[rt])

        print(path_coords)
        print(path_idx)
        return path_coords, path_idx

    def coords_to_idx(self, coords):
        '''
        coords: 2D array of coordinates where each row is [t,x,y,z]

        Returns idx: a 2D array of indices [t, nx, ny, nz] corresponding to input 
                    coordinates, if coordinates are inbetween index values the function rounds down
        '''
        idx = (coords - [self.t_min, self.x_min, self.y_min, self.z_min])/[self.dt, self.dx, self.dy, self.dz]
        return np.floor(idx).astype('int')
    
    def idx_to_coords(self, idx):
        '''
        Parameters
        ----------
        idx : 2D array
            an array of indices where each row is [nt,nx,ny,nz]

        Returns
        -------
        coords: 2D array
            an array of coordinates [t, x, y,z] corresponding to input indices
        '''
        return idx*[self.dt, self.dx, self.dy, self.dz] + self.origin
        
if __name__ == '__main__':
    # from utils import load_yaml

    # mpc_params = load_yaml('sim_params.yaml')
    # planner_params = mpc_params['planner']

    # planner = Planner(**planner_params)
    planner = Planner(x_min=Config.x_min, x_max=Config.x_max, y_min=Config.y_min, y_max=Config.y_max, z_min=Config.z_min, z_max=Config.z_max, dx=Config.dx, dy=Config.dy, dz=Config.dz, dt=Config.dt, horizon=Config.horizon, window=Config.window, 
                 j_t=Config.j_t, j_s=Config.j_s, obs_padding=Config.obs_padding, d_replan=Config.d_replan, v_replan=Config.v_replan, t_replan=Config.t_replan)
    # target_state = np.array([[1, 1, -0.3, 0.]]) #agjust to [x,y,z,vx,vy,vz]
    target_state = np.array([[1.0, 0.5, 1.0, 0.0, 0.0, 0.0]]) #agjust to [x,y,z,vx,vy,vz]
    # obstacle_state = np.array([[-4.2, 1.2, 0, 0., 1.],\
    #                           [-2, -1.8, 0, 0., 1.],\
    #                           [-1.7, -3.5, 0, 0., 0.7],\
    #                           [-0.3, 3, 0, 0., 1.],\
    #                           [2.1, 2, 0, 0., 0.5],\
    #                           [3.9, -2.5, 0, 0., 1.4]]) #adjust to [x,y,z,h,r,vx,vy,vz]
    obs_init = np.zeros((7,1)) 
    obs_init[:5,0] = -1.8, -1.8,    2, 0.3, 2 
    obstacle_state = obs_init.T
    agent_state = np.array([0,-5,1,0,0.0,0]) #adjust to [x,y,z,vz,vy,vz]

    paths, pidx = planner.plan(0.05, None, agent_state, target_state, obstacle_state)
    # c = planner.cost

    # vis = PlanVisualizer(planner)
    # vis.set_path(paths[:,0,:])


    '''
    pseudocode (modified from Jacky's 2D path planner into 3D)
    
    target_state : array of type [x,y,z,vx,vy.vz]
    obstacle_state : Nx5 array, each row contains [x,y,z,h,r,vx,vy].T for the nth 
    t_range : array of type [t_max, dt]
    x_range : array of type [x_min, x_max, dx]
    y_range : array of type [y_min, y_max, dy]
    z_range : array of type [z_min, z_max, dz]

    1) get dimensions of the 4d grid, Nt = t_max/dt, Nx = (x_max-x_min)/dx, Ny = (y_max-y_min)/dy, Nz = (z_max-z_min)/dz 
    2) generate a 4d grid of size [nt,nx,ny,nz] and initialize to np.inf, this is the cost grid
       the origin of each 2d layer is at (x_min,y_min,z_min)
       keep indexing [t,x,y,z]. Even though arrays index row first, it shouldn't affect the calculations

    3) step through the target and obstacles in time:
        for the target:
            get the cells it occupies through time and assign the cost[nt,nx,ny,nz] = 0 at those cells
            
        for the obstacle:
            get the cells it occupies through time and assign the cost[nt,nx,ny,nz] = -1 at those cells
            we just need a way to disambiguate occupied cells and cells not searched
            
    4) run BFS on the last layer, i.e. bfs(cost[Nt-1]), ignoring obstacle cells
    
    5) run min pooling back in time, but a few modifications
        
        a) we require a cost per time step: c_t
        b) we only update unoccupied cells, which corresponds to cells with cost > 0
        c) don't penalize distance cost if we don't move
            wait_cost = cost[t] + c_t
    
        min pool:
            cost[t][cost[t] == -1] = np.inf

            cost[t-1][cost[t-1] > 0] = min_pool(cost[t], kernel=(3,3), padding=inf) + 1 + c_t

        check if it's better to wait:
            cost[t-1][cost[t-1] > 0] = min(cost[t-1][cost[t-1] > 0], wait_cost[cost[t-1]>0])
            
    '''
    
    
    
    
