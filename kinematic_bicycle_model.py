import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

#PARAMETERS
WHEEL_BASE      = 2.5
LR              = 1.5
LF              = WHEEL_BASE - LR
STEER_RATIO     = 19.5
MAX_STEER_DELTA = math.radians(45.0) #

#plot car parameters
LENGTH       = 3.5  # [m]
WIDTH        = 1.5  # [m]
BACKTOWHEEL  = 0.5  # [m]
WHEEL_LEN    = 0.8  # [m]
WHEEL_WIDTH  = 0.25  # [m]
TREAD        = 0.7  # [m]
WB           = WHEEL_BASE 


def plot_car_cg(x, y, yaw, steer=0.0, cabcolor="w", truckcolor="k"):

    outline = np.matrix([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                         [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    outline = np.matrix([[-(LR + BACKTOWHEEL), (LENGTH - BACKTOWHEEL - LR), (LENGTH - BACKTOWHEEL - LR), -(LR + BACKTOWHEEL), -(LR + BACKTOWHEEL)],
                         [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.matrix([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                          [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)
    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.matrix([[math.cos(yaw), math.sin(yaw)],
                      [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.matrix([[math.cos(steer), math.sin(steer)],
                      [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T * Rot2).T
    fl_wheel = (fl_wheel.T * Rot2).T

    fr_wheel[0, :] += (WB - LR)
    fl_wheel[0, :] += (WB - LR)

    rr_wheel[0, :] += -LR
    rl_wheel[0, :] += -LR

    fr_wheel = (fr_wheel.T * Rot1).T
    fl_wheel = (fl_wheel.T * Rot1).T

    outline = (outline.T * Rot1).T
    rr_wheel = (rr_wheel.T * Rot1).T
    rl_wheel = (rl_wheel.T * Rot1).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.fill(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten())
    
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)

    plt.fill(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), 'k')
    
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)

    plt.fill(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), 'k')

    
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)

    plt.fill(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), 'k')
    
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)

    plt.fill(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), 'k')
    
    plt.plot(x, y, "r*") 
    return None
    

def plot_trajectory(x,y,theta,delta, v, dt = 0.2):
    for i in range(0, len(x)):
        plt.cla()
        plt.plot(x, y, "-g", label="trajectory")
        plt.axis('equal')
        plt.grid(True)
        plot_car_cg(x[i], y[i], theta[i], delta[i])
        plt.axis('equal')
        plt.pause(0.1)
    plt.show()
    return None

def plot_curvature(data):
    dx  = np.gradient(data['x'].flatten())
    dy  = np.gradient(data['y'].flatten())
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5
    radius = 1.0 /curvature
    print(' Minimum radius is ' + str(min(abs(radius))))
    #plt.plot(range(len(radius)), radius, 'g', label = 'radius')

    plt.plot(data['x'].flatten(), data['y'].flatten(), 'g', label = 'radius')
    plt.show()
    plt.plot(dx,dy, 'g', label = 'radius')
    plt.show()
    plt.plot(range(len(radius)), radius, 'g', label = 'radius')
    plt.show()

    return radius
    
class Bicycle_model():
    def __init__(self, wheel_base = 2.5, max_wheel_angle = np.radians(41.0), dt = 0.2,
                 st_grad = np.radians(10.0)):
        """Creates robot and initializes location/orientation to 0, 0, 0."""
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0
        
        self.wb = 2.5
        self.lr = 1.5    
        self.sample_time = 0.2

        self.steering_noise   = 0.0
        self.distance_noise   = 0.0
        self.steering_drift   = 0.0
        self.max_wheel_angle  = max_wheel_angle
        self.steer_grad_limit = st_grad

    def set(self, x, y, theta):
        """Sets a robot coordinate."""
        self.xc = x
        self.yc = y
        self.theta = theta % (2.0 * np.pi)
        
    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

    def set_noise(self, steering_noise, distance_noise):
        """ Sets the noise parameters."""
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.steering_noise = steering_noise
        self.distance_noise = distance_noise

    def set_steering_drift(self, drift):
        """ Sets the systematical steering drift parameter """
        self.steering_drift = drift

    def status_check(self):
        print('entered')
        if self.delta > self.max_wheel_angle:
            self.delta = self.max_wheel_angle
        if self.delta < -self.max_wheel_angle:
            self.delta = -self.max_wheel_angle
        self.theta = self.theta % (2.0 * np.pi)
        print('done')

    def move(self, v, wh_ang_dem):
        self.status_check()
        self.xc    = self.xc    + self.sample_time * v * np.cos(self.beta + self.theta)
        self.yc    = self.yc    + self.sample_time * v * np.sin(self.beta + self.theta)
        self.theta = self.theta + self.sample_time *  ((v * np.cos(self.beta) * np.tan(self.delta))/ self.wb)

        delta_grad = (wh_ang_dem - self.delta) / self.sample_time        

        if delta_grad > self.steer_grad_limit:
            delta_grad = self.steer_grad_limit
            
        if delta_grad < -self.steer_grad_limit:
            delta_grad = -self.steer_grad_limit
                                
        self.delta = self.delta + self.sample_time *  delta_grad
        self.status_check()
        self.beta  = np.arctan2((np.tan(self.delta) * self.lr * self.wb), 1)
        pass

    def __repr__(self):
        return '[x=%.5f y=%.5f orient=%.5f delta=%.5f ]' % (self.x, self.y, self.orientation, self.delta)


def initialise_noise_params():
    noise_params = {}
    noise_params['STEERING_NOISE'] = np.radians(0.0)
    noise_params['STEERING_DRIFT'] = np.radians(0.0)
    noise_params['DISTANCE_NOISE'] = 0.20
    return noise_params

# run - does a single control run
def make_robot( init_pose, Noise_params):
    """ Resets the robot back to the initial position and drift.
    You'll want to call this after you call `run`. """
    robot = Bicycle_model()
    robot.set(init_pose[0], init_pose[1], init_pose[2])
    robot.set_noise( Noise_params['STEERING_NOISE'], Noise_params['DISTANCE_NOISE'])
    robot.set_steering_drift(Noise_params['STEERING_DRIFT'])
    return robot


# NOTE: We use params instead of tau_p, tau_d, tau_i
def run_trial(robot, params, TEST, n=450, speed=1.5):
    trial_data = {}
    x_trajectory  = np.zeros((n,1))
    y_trajectory  = np.zeros((n,1))
    orientaion_tr = np.zeros((n,1))
    delta_tr      = np.zeros((n,1))
    cte_tr        = np.zeros((n,1))
    psi_err       = np.zeros((n,1)) 

    if TEST =='Lat_offset':
        prev_cte = 0.0
        int_cte = 0
        for i in range(n):
            cte = robot.yc
            psi_error = robot.theta
            cte_grad = cte - prev_cte
            steer_dem = -params['kp_lat']*cte - params['kd_lat']*cte_grad - params['ki_lat']*int_cte
            robot.move(speed, steer_dem)
            #resetting errors
            int_cte += cte
            prev_cte = cte

            #storing data for plotting
            x_trajectory[i]  = robot.xc
            y_trajectory[i]  = robot.yc
            orientaion_tr[i] = robot.theta
            delta_tr[i]      = robot.delta
            cte_tr[i]        = cte
            psi_err[i]       = psi_error

    if TEST =='Head_offset':
        prev_psi_err = 0.0
        int_psi_err  = 0.0
        for i in range(n):
            cte = robot.yc
            psi_error = robot.theta
            psi_grad = psi_error - prev_psi_err
            steer_dem = -params['kp_head']*psi_error - params['kd_head']*psi_grad - params['ki_head']*int_psi_err
            robot.move(speed, steer_dem)
            #resetting errors
            int_psi_err += psi_error
            prev_psi_err = psi_error

            #storing data for plotting
            x_trajectory[i]  = robot.xc
            y_trajectory[i]  = robot.yc
            orientaion_tr[i] = robot.theta
            delta_tr[i]      = robot.delta
            cte_tr[i]        = cte
            psi_err[i]       = psi_error
        
    trial_data['x']          = x_trajectory
    trial_data['y']          = y_trajectory
    trial_data['theta']      = orientaion_tr
    trial_data['delta']      = delta_tr
    trial_data['cte']        = cte_tr
    trial_data['epsi_error'] = psi_err
    return trial_data


''' sample test with left circle of radius 10.0 m '''

'''
sample_time = 0.2
time_end = 20
model = Bicycle_model()

# set delta directly to atan(wheelbase / radius)
model.delta = np.arctan(2.5/1.0)

#variables for plotting
t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
theta_data = np.zeros_like(t_data)
delta_data = np.zeros_like(t_data)
cg_to_ra = model.lr
x_ra = np.zeros_like(t_data)
y_ra = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    theta_data[i] = model.theta
    delta_data[i] = model.delta
    x_ra[i] = x_data[i] - cg_to_ra * np.cos(theta_data[i])  
    y_ra[i] = y_data[i] - cg_to_ra * np.sin(theta_data[i])  
    model.move(np.pi, np.arctan(2.5/1.0) )

plot_trajectory(x_data, y_data, theta_data, delta_data, [1.5] * len(x_data), dt = 0.1)
##plt.axis('equal')
##plt.plot(x_data, y_data,label='CG')
####plt.plot(x_ra, y_ra,label='Rear axle')
##plt.legend()
##plt.show()
'''

''' parallel lateral offset test '''
#initialise noise parameters
noise_params = initialise_noise_params()

gain_params= {}
gain_params['kp_lat'] = 0.4
gain_params['kd_lat'] = 0.5
gain_params['ki_lat'] = 0.0
gain_params['kp_head'] = 0.5
gain_params['kd_head'] = 0.1
gain_params['ki_head'] = 0.0


#make an instance of bicycle robot model with initial pose[x,y,theta] and noise parameters
robot    = make_robot([0.0, 1.0, np.radians(0.0)], noise_params)
data     = run_trial(robot, gain_params, 'Lat_offset', speed=1.5)
rad_data =  plot_curvature(data)


#make an instance of bicycle robot model with initial pose[x,y,theta] and noise parameters
#robot    = make_robot([0.0, 0.0, np.radians(20.0)], noise_params)
#data     = run_trial(robot, gain_params, 'Head_offset', speed=1.5)
#rad_data =  plot_curvature(data)

plot_trajectory(data['x'], data['y'], data['theta'], data['delta'], [1.5] * len(data['delta']), dt = 0.2)

##n = len(data['x'])
##fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))
##ax1.plot(data['x'], data['y'], 'g', label='PID controller')
##ax1.plot(data['x'], np.zeros(n), 'r', label='reference')
##ax2.plot(range(len(data['theta'])), np.degrees(data['theta']), 'c', label='reference')
##plt.axis('equal')
plt.show()
