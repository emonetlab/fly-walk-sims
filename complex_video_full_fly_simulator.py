# Written by Hope Anderson
# Emonet Lab, June 2019
# Advised by Nirag Kadakia
# Edited by Viraaj Jayaram July 2020

import scipy as sp
from load_videos import load_vid_by_frm

class VectorFlies:
    
    ##################### INSTANCE VARIABLES #####################
    #
    # CONSTANTS
    # delta_t: the amount of time in seconds for one timestep
    # delta_pos: change in position in mm between two consecutive
    #     position elements
    # speed: walking speed of the flies in mm/s
    # num_steps: the number of steps to execute
    # num_flies: the number of flies to run
    # px_per_step: the number of pixels a fly can cover in one step
    # big_arena: if True, use full smoke video (315.4 x 184.8 mm). If False, truncate arena by
    #     30 mm horizontally and 35 mm vertically on each side, resulting in 267.7 x 114.0 mm arena.
    # bck_sub: if 1, use background subtraction on smoke video; if 0, no background subtraction.
    # real_antenna: if True, use elliptical fly antenna centered 2.3 mm in front of fly position.
    #     If False, use square arena centered about fly position. These antennas are of comparable
    #     sizes.
    #
    # MODEL PARAMETERS
    # Walk-to-stop parameters
    # ws_lambda_0: base rate without any whiffs. Units = Hz
    # ws_del_lambda: scale factor for whiff rate modification. Units = Hz
    # ws_tau_R: timescale for whiff rate modification decay (dependent only on last
    #        whiff). Units = seconds
    #
    # Stop-to-walk parameters
    # sw_lambda_0: base rate without any whiffs. Units = Hz
    # sw_del_lambda: scale factor for whiff rate modification. Units = Hz
    # sw_tau_H: timescale for whiff rate modification decay (accumulative).
    #        Units = seconds
    #
    # Turn parameters
    # turn_lambda: number of turns that occur per second.
    # turn_alpha: scale factor for left turn probability.
    # tau_turn: timescale for whiff modification to left turn probability
    #        (accumulative). Units = seconds.
    # turn_mean: Average degrees a fly turns during a left turn.
    # turn_std: Standard deviation of fly turning distribution. Units = 1/timestep
    # no_turn_mean: Average change in body angle degrees when fly is not turning.
    # no_turn_std: Standard deviation of body angle changes when not turning.
    #        Units = 1/timestep.
    # base_upturn_rate: the fraction of turns that are upwind without any
    #        whiff modulation. Note that when using a sigmoidal turn bias this is automatically 
    #        set to 0.5 by the sigmoid and this variable is not actually used to compute the probability. 
    #        One should manually change the _turnLprob() method if one wants to change this. 
    #
    # RANDOM NUMBERS
    # Random number generator is seeded by the following integer: 10000 * job + num_flies.
    # ranfs: a list of uniformly random floats between 0 and 1.
    # randns: a list of random floats from a Gaussian distribution centered at 0 with standard deviation 1.
    # ranf_count: the index of the next unused random number in ranfs. Each time a ranf is accessed.
    #        this should be incremented so no number is used twice. Ranfs should be long enough that 
    #        ranf_count will never go out of bounds.
    # randn_count: index of next unused random number in randns; follows same criteria as ranf_count.
    #
    # SAVE DATA INFO
    # save_data: if True, the transpose of the following instance variables are 
    #        saved: x, y, theta, walks, turns, whiff_whiffs, wt. Thus the first dimension corresponds to fly
    #        number and the second to time point.
    # data_dir: the directory in which to save the data.
    # job: the job number of this instance of VectorFlies.
    #
    # INITIAL POSITION DATA
    # time: array of timepoints for simulation for each fly. Starts at start_t
    #     and then increments by delta_t at every step.
    # x: array of the x-positions of each fly in mm (fly position x fly number). If start_from_file
    #     is True, the intial locations are taken directly from x_file. Otherwise the x_positions
    #     are randomly generated in the interval given by start_x_pos.
    # y: array of the y-positions of each fly in mm (fly position x fly number); same initialization
    #     criteria as x.
    # theta: array of the body angles of each fly in time. Units = degrees.
    #     0 degrees is downwind, 180 is upwind. (fly theta x fly number); same initialization criteria
    #     as x.
    #
    # v_x: array of the x-velocity values of each fly over time (fly velocity x fly number)
    # v_y: array of the y-velocity values of each fly over time (fly velocity x fly number)
    # del_theta: array of the changes in theta values over time Units = degrees.
    #      (fly del theta x fly number)
    #
    # x_idx: the current x indices in the signal array
    #      (x_idx index = fly number). Determined from rounding x / mm_per_px to nearest integer.
    # y_idx: the current y indices in the signal array
    #      (y_idx index = fly number). Determined similarly to x_idx.
    #
    # SIGNAL LOCATION
    # subdir: the subdirectory in which to find the signal data
    # file: the file in which to find the signal data
    # assay_type: the assay type of the signal data
    #
    # SIGNAL/WHIFF DATA
    # signal_idxs: the time indices of the signal array to load.
    # signal: a two-dimmensional binary array of signal at current timepoint.
    #      The first dimension is the x index; the second is the y index.
    #      Size is assumed to be 2048x1200.
    # whiff_threshold: the minimum amount of time between whiffs.
    # whiff_min: the minimum signal value that registers as a whiff.
    #
    # antenna_off: the offset from the actual fly position index from which the
    #       antenna is calculated. 
    #
    # antenna_sig; the average signal reading over antenna area for each fly.
    #
    # whiff_hits: will track all whiff encounters for each fly. 1 when whiff is
    #      encountered, 0 when no whiff. (num_steps x num_flies)
    #
    # hits_window: an array of whiff hits from the last 500 time indices for each fly.
    #       Before the 499th timestep, the second dimension of this array has length
    #       of the number of timesteps.
    #
    # last_switch: the time step of the last stop/walk switch of each fly. Used for
    #      tracking the whiffs since this switch.
    #
    # wt: an array of the accumulative whiff function used for turn probabilities.
    #       Calculated using the hits_window whiffs.
    #
    # last_whiff: a vector of the times of the last whiff encountered for each fly.
    #      Initialized to -whiff_threshold so that the first whiff will always count.
    #
    # encountered_low: Elements are true True if the corresponding fly encountered
    #       a low signal region since the last whiff hit,
    #       False otherwise. In order to register a new whiff, encountered_low must be
    #       True.
    #
    # STATE (WALK, STOP, TURN) INFORMATION
    # wall_behavior: string that describes how flies will behave when encountering a wall
    #       of the area. If 'turn', the flies will turn away from wall at random angle. 
    #       If 'walk', flies will continue as if there are no walls, but they will
    #       not register any whiffs outside of the arena. If 'stop', the flies just
    #       stop indefinitely at the wall. 'Stop' behavior is untested.
    #
    # hit_wall: True for flies that have hit the wall, False for those that haven't
    #
    # wall_hits: 1 when there is a wall hit for fly at given time point, 0 if no wall hit.
    #
    # is_walking: Element is True when given fly is walking, else False
    # walks: binary hits array for walks for each fly. 1 when walking, 0 if not.
    # stops: binary hits array for stops for each fly. 1 when stopped, 0 if not.
    # turns: binary hits array of turns for each fly. 1 when turning, 0 when not.
    #
    # sw_fn: array of stop-to-walk rate at every step for each fly. 0 when fly is walking.
    # ws_fn: array of walk-to-stop rate at every step for each fly. 0 when fly is stopped.
    #
    # VELOCITY DATA
    # vx: x-component of fly velocities over all time
    # vy: y-component of fly velocities over all time
    # del_theta: change in theta of all flies over all time
    # theta_correct: theta corrections made by wall behavior corrections over all flies, all time.
    #
    # COUNTER
    # i: the current step; used for array indexing
    #
    #################################### METHODS ####################################
    #
    # An underscore before the method name indicates that the method should
    # only be used internally.
    #
    # go(self, num_steps):
    #    loads the signal array and then calls the step method for each time step.
    #    args:
    #         num_steps: the total number of steps for which the simulation will run.
    #     
    #
    # step(self):
    #    executes one step in the simulation. Updates velocity and position data,
    #    determines whether fly will stop or walk at next step, and registers
    #    whiff information.
    #
    # _ws_rate(self):
    #    calculates the walk to stop rate for a given step.
    #    returns:
    #         the walk-to-stop rate of this step.
    #
    # _sw_rate(self):
    #    calculates the stop to walk rate for a given step.
    #    returns:
    #         the stop-to-walk rate of this step.
    #
    # _turnL_prob(self):
    #    calculates the probability of the fly turning left, given that this step
    #    invovles a turn.
    #    returns:
    #         the probability of turning left at this step.
    #
    # _wt(self):
    #    calculates and stores the value of the wt instance variable for a given step.
    #
    # _happen(self, prob):
    #    Determines whether an event will happen via Monte Carlo random number generation.   
    #    args:
    #         prob: the probability that an event will happen.
    #    returns:
    #         True if the event will happen, False if not.
    #
    # _poisson_happen(self, rate):
    #    Determines the probability of a Poisson event happening at this time point
    #    and calls _happen to see if the event does happen.
    #    args:
    #         rate: the rate of the Poisson process events in Hz. Must be a float or integer.
    #    returns:
    #         True if event happens, False if not.
    #
    # make_antenna(self, fly_x_idx, fly_y_idx, fly_theta, r_from_fly_center=14, a=5, b=1.5):
    #    Returns the indices of an elliptical antenna for a given fly location and orientation.
    #    The antenna is located a distance r_from_fly_center from the fly x and y coordinates.
    #    It has semimajor axis a and semiminor axis b. The semimajor axis is oriented perpindicular
    #    to the fly orientation angle.
    #    args:
    #         fly_x_idx: the x-index of the fly in the signal array
    #         fly_y_idx: the y-index of the fly in the signal array
    #         fly_theta: the fly's orientation
    #         r_from_fly_center: the distance in pixels between the center of the fly and
    #              the center of the antenna.
    #         a: semimajor axis of antenna in pixels
    #         b: semiminor axis of antenna in pixels
    #     returns:
    #         a tuple of (x_idxs of antenna, y_idxs of antenna).
    #
    ############################### CONSTRUCTOR ARGS ################################
    #
    # init_theta: a pair giving the range of the initial body angles of the flies.
    # start_x_pos: a pair giving the range of the initial x-positions of the flies.
    # start_y_pos: a pair giving the range of the initial y-positions of the fly.
    # signal_idxs: the time indices of the signal array to load. Length should be greater
    #        than or equal to the number of time steps.
    # num_flies: the number of flies to run
    # num_steps: the number of timesteps to execute.
    #
    # OPTIONAL ARGS
    # init_is_walking: True if fly is walking initially, otherwise False. Default True.
    # start_t: the starting time. Defaults to 0 seconds.
    # signal_mean: the mean reading of the signal array. (Signal intensities are assumed to
    #        follow a normal distribution).
    # signal_std: the standard deviation of the signal intensity distribution.
    # whiff_sigma_coeff: Whiffs must have an intensity of at least signal_std * whiff_sigma_coeff
    #        past the mean. Default to 3.0.
    # antenna_len: if real_antenna is False, then this gives the side length of the virtual antenna of 
    #        the fly in indices. Default to 5. MUST BE AN INTEGER.
    # job: the job number of this run.
    # start_from_file: if True, initial positions and orientations are drawn from pickle files
    #        init_xs, init_ys, and init_thetas
    # start_dir: the directory in which the start files are located; only used if start_from_file
    #        is True.
    # Other optional args correspond directly to the instance variables of the same name.
    #
    def __init__(self, init_theta, start_x_pos, start_y_pos, signal_idxs, num_steps, num_flies,
                 job=0, init_is_walking = True, start_t=0,
                 speed = 11.5, delta_t=1/89.94, delta_pos=0.154, whiff_threshold = 0.1, signal_mean = -0.139,
                 signal_std = 1.08, whiff_sigma_coeff = 2.5, antenna_len=5, ws_lambda_0 = 0.78,
                 ws_del_lambda = -0.61, ws_tau_R = 0.25, sw_lambda_0 = 0.29, sw_del_lambda = 0.41,
                 sw_tau_H = 0.52, turn_lambda = 1/0.75, turn_alpha = 0.242, tau_turn = 2.0, turn_mean = 30.0,
                 turn_std = 8, no_turn_mean = 0, no_turn_std = 20/89.94,
                 subdir="../../../../data", wall_behavior = 'turn', base_upturn_rate=0.5,
                 file="2018_09_12_NA_3_3ds_5do_IS_1-frames.mat", bck_file = "2018_09_12_NA_3_3ds_5do_IS_1.mat", assay_type='IS', save_data=True, data_dir=".",
                 start_from_file=False,start_dir='',big_arena=False,bck_sub=0, real_antenna=False):
 
        ############### CONSTANTS ###############

        self.delta_t = delta_t
        self.delta_pos = delta_pos
        self.speed = speed
        self.num_steps = num_steps
        self.num_flies = num_flies
        self.px_per_step = sp.ceil(self.speed / self.delta_pos * self.delta_t)
        self.big_arena=big_arena
        self.bck_sub = bck_sub
        self.real_antenna = real_antenna

        ########## MODEL PARAMETERS ###########

        # walk-to-stop parameters
        self.ws_lambda_0 = ws_lambda_0
        self.ws_del_lambda = ws_del_lambda
        self.ws_tau_R = ws_tau_R

        # stop-to-walk parameters
        self.sw_lambda_0 = sw_lambda_0
        self.sw_del_lambda = sw_del_lambda
        self.sw_tau_H = sw_tau_H
        
        # turn parameters
        self.turn_lambda = turn_lambda
        self.turn_alpha = turn_alpha
        self.tau_turn = tau_turn
        self.turn_mean = turn_mean
        self.turn_std = turn_std
        self.no_turn_mean = no_turn_mean
        self.no_turn_std = no_turn_std
        self.base_upturn_rate = base_upturn_rate

        ############# RANDOM NUMBERS ##############
        rand_gen = sp.random.RandomState(10500 * int(job) + int(num_flies))
        self.ranfs = rand_gen.random_sample(size=(self.num_flies * self.num_steps * 5))
        self.randns = rand_gen.normal(size=(self.num_flies * self.num_steps))
        self.ranf_count = 0
        self.randn_count = 0

        ########### SAVE DATA INFO ##############
        self.save_data = save_data
        self.data_dir = data_dir
        self.job = job

        ########### INITIAL POSITION DATA #############
        
        self.time = sp.zeros((self.num_steps, self.num_flies))
        self.time[0] = float(start_t)

        self.x = sp.zeros((self.num_steps, self.num_flies))
        
        if(start_from_file):
            x_file = open(str(start_dir) + 'init_xs','rb')
            self.x[0] = sp.array(pkl.load(x_file))
            x_file.close()

        else:
            start_x = start_x_pos[0]
            start_x_range = start_x_pos[1] - start_x_pos[0]
        
            for i in range(self.num_flies):
                self.x[0,i] = start_x + start_x_range * rand_gen.random_sample()

        self.y = sp.zeros((self.num_steps, self.num_flies))

        if(start_from_file):
            y_file = open(str(start_dir) + 'init_ys','rb')
            self.y[0] = sp.array(pkl.load(y_file))
            y_file.close()

        else:
            start_y = start_y_pos[0]
            start_y_range = start_y_pos[1] - start_y_pos[0]
            
            for i in range(self.num_flies):
                self.y[0,i] = start_y + start_y_range * rand_gen.random_sample()

        self.theta = sp.zeros((self.num_steps, self.num_flies))
        
        if(start_from_file):
            theta_file = open(str(start_dir) + 'init_thetas','rb')
            self.theta[0] = sp.array(pkl.load(theta_file))
            theta_file.close()
            
        else:
            start_theta = init_theta[0]
            start_theta_range = init_theta[1] - init_theta[0]
            
            for i in range(self.num_flies):
                self.theta[0,i] = (start_theta + start_theta_range
                                   * rand_gen.random_sample())
                
        self.x_idx = sp.zeros(self.num_flies, dtype=int)
        self.x_idx = sp.array(sp.rint(self.x[0] / delta_pos), dtype=int)

        self.y_idx = sp.zeros(self.num_flies, dtype=int)
        self.y_idx = sp.array(sp.rint(self.y[0] / delta_pos), dtype=int)

        ############ SIGNAL LOCATION ############
        self.subdir = subdir
        self.file = file
        self.bck_file = bck_file
        self.assay_type = assay_type
        
        ########### SIGNAL/WHIFF DATA ###########
        
        self.signal_idxs = signal_idxs

        self.whiff_threshold = whiff_threshold
        self.whiff_min = signal_mean + whiff_sigma_coeff * signal_std
        
        # initialized so that the first hit will always count
        self.last_whiff = sp.full_like(self.num_flies, -whiff_threshold) 
        
        self.antenna_off = antenna_len//2

        self.antenna_sig = sp.zeros(self.num_flies)

        self.whiff_hits = sp.zeros((self.num_steps, self.num_flies))

        self.hits_window = sp.array([self.whiff_hits[0]])
        
        self.last_switch = sp.zeros(self.num_flies, dtype=int)

        self.wt = sp.zeros((self.num_steps, self.num_flies))

        self.last_whiff = sp.full(self.num_flies, -self.whiff_threshold)

        self.encountered_low = sp.full(self.num_flies, 1.0)
    
        ########## STATE INFORMATION ##########
        if(wall_behavior != 'turn' and wall_behavior != 'stop' and wall_behavior != 'walk'):
            raise ArgError("Wall behavior argument must be \"turn\", \"stop\", or \"walk\". Found: "
                           + str(wall_behavior))

        self.wall_behavior = wall_behavior

        self.hit_wall = sp.zeros(self.num_flies)

        self.wall_hits = sp.zeros((self.num_steps, self.num_flies))
        
        self.is_walking = sp.full(self.num_flies, init_is_walking)

        self.walks = sp.zeros((self.num_steps, self.num_flies))

        self.stops = sp.zeros((self.num_steps, self.num_flies))

        self.turns = sp.zeros((self.num_steps, self.num_flies))
        
        self.ws_fn = sp.zeros((self.num_steps, self.num_flies))

        self.sw_fn = sp.zeros((self.num_steps, self.num_flies))

        ############ INITIAL VELOCITY DATA #################
        self.vx = sp.zeros((self.num_steps, self.num_flies))

        self.vy = sp.zeros((self.num_steps, self.num_flies))

        self.del_theta = sp.zeros((self.num_steps, self.num_flies))

        self.theta_correct = sp.zeros((self.num_steps, self.num_flies))
        
    def go(self):
        for i in range(self.num_steps):
            self.i = i
            signal_idx = self.signal_idxs[i]
                        
            if(self.big_arena):
                self.signal = load_vid_by_frm(subdir=self.subdir, file=self.file, 
                                              frame=signal_idx, bck_sub=self.bck_sub, bck_file = self.bck_file)
            else:
                self.signal = load_vid_by_frm(subdir=self.subdir, file=self.file, 
                                              frame=signal_idx, bck_sub=self.bck_sub)[200:1938,230:970]


            self.step()


        if(self.save_data):

            job_str = "Job" + str(self.job) + "_"
            x_trans = sp.transpose(self.x)
            y_trans = sp.transpose(self.y)
            theta_trans = sp.transpose(self.theta)
            walks_trans = sp.transpose(self.walks)
            turns_trans = sp.transpose(self.turns)
            hits_trans = sp.transpose(self.whiff_hits)
            wt_trans = sp.transpose(self.wt)
            
            sp.savetxt(job_str + "xs", x_trans)
            sp.savetxt(job_str + "ys", y_trans)
            sp.savetxt(job_str + "thetas", theta_trans)
            sp.savetxt(job_str + "walks", walks_trans)
            sp.savetxt(job_str + "turns", turns_trans)
            sp.savetxt(job_str + "whiffs", hits_trans)
            sp.savetxt(job_str + "wts", wt_trans)
        
    def step(self):

        # determine whether a whiff occurs here; update instance variables accordingly

        for i in range(self.num_flies):
            x_in_range = (self.x_idx[i] >= 0) and (self.x_idx[i] < len(self.signal))
            y_in_range = (self.y_idx[i] >= 0) and (self.y_idx[i] < len(self.signal[0]))

            if(x_in_range and y_in_range):
                if(not self.real_antenna):
                    signal_slice = self.signal[max(self.x_idx[i]-self.antenna_off,0):
                                               min(self.x_idx[i]+self.antenna_off+1,len(self.signal)),
                                               max(self.y_idx[i] - self.antenna_off,0):
                                               min(self.y_idx[i] + self.antenna_off + 1,len(self.signal[0]))]

                    self.antenna_sig[i] = sp.mean(signal_slice)

                else:
                    fly_x_idx = self.x_idx[i]
                    fly_y_idx = self.y_idx[i]
                    fly_theta = self.theta[self.i,i]

                    antenna_slice = self.make_antenna(fly_x_idx,fly_y_idx,fly_theta)

                    # if antenna is out of bounds, set signal reading to 0
                    if(sp.any(antenna_slice[0] >= len(self.signal)) or sp.any(antenna_slice[1] >= len(self.signal[1]))):
                        self.antenna_sig[i] = 0.0

                    else:
                        signal_slice = self.signal[antenna_slice]
                        self.antenna_sig[i] = sp.mean(signal_slice)


            else:
                self.antenna_sig[i] = 0.0

        high_sig_idxs = sp.where(self.antenna_sig >= self.whiff_min)[0]

        new_high_idxs = sp.intersect1d(high_sig_idxs,
                                       sp.where(self.encountered_low)[0])

        whiff_idxs = sp.intersect1d(new_high_idxs,
                                    sp.where((self.time[self.i,:] - self.last_whiff) > self.whiff_threshold)[0])

        not_whiff_idxs = sp.setdiff1d(range(self.num_flies), whiff_idxs)

        low_sig_idxs = sp.where(self.antenna_sig < self.whiff_min)[0]

        self.encountered_low[low_sig_idxs] = 1.0
        self.encountered_low[high_sig_idxs] = 0.0

        self.last_whiff[whiff_idxs] = self.time[self.i,0]
        self.whiff_hits[self.i, whiff_idxs] = 1.0
        self.whiff_hits[self.i, not_whiff_idxs] = 0.0

        # keep hits window at constant length
        if(len(self.hits_window) < 500):
            self.hits_window = self.whiff_hits[:self.i+1]
        else:
            self.hits_window = self.whiff_hits[self.i-500:self.i+1]
                
        # update sw_fn, ws_fn
        self.ws_fn[self.i] = self._ws_rate() * self.is_walking
        self.sw_fn[self.i] = self._sw_rate() * sp.logical_not(self.is_walking)
        
        # determine wt for this step
        self._wt()
        
        # make sure angle is between 0 and 359 degrees
        for fly in range(self.num_flies):
            if (self.theta[self.i, fly] // 360. != 0):
                self.theta_correct[self.i, fly] = 1.0
                self.theta[self.i,fly] = self.theta[self.i,fly] % 360.0
        
        #
        # make sure flies won't hit a wall
        #
        if(self.wall_behavior == 'turn'):
            wall_risk_ll = sp.intersect1d(sp.where(self.x_idx <= self.px_per_step)[0],
                                          sp.where(self.y_idx <= self.px_per_step)[0])
            
            for ind in wall_risk_ll:
                self.wall_hits[self.i, ind] = 1.0
                self.hit_wall[ind] = 1.0
                self.theta[self.i, ind] = self.ranfs[self.ranf_count] * 90.0
                self.ranf_count = self.ranf_count + 1
                
            wall_risk_ul = sp.intersect1d(sp.where(self.x_idx <= self.px_per_step)[0],
                                          sp.where(self.y_idx >= (len(self.signal[0]) - self.px_per_step - 2))[0])
                
            for ind in wall_risk_ul:
                self.wall_hits[self.i, ind] = 1.0
                self.hit_wall[ind] = 1.0
                self.theta[self.i, ind] = self.ranfs[self.ranf_count] * 90.0 + 270.0
                self.ranf_count = self.ranf_count + 1
                
            wall_risk_lr = sp.intersect1d(sp.where(self.x_idx >= (len(self.signal) - self.px_per_step - 2))[0],
                                          sp.where(self.y_idx <= self.px_per_step)[0])
                
            for ind in wall_risk_lr:
                self.wall_hits[self.i, ind] = 1.0
                self.hit_wall[ind] = 1.0
                self.theta[self.i, ind] = self.ranfs[self.ranf_count] * 90.0 + 90.0
                self.ranf_count = self.ranf_count + 1
                
            wall_risk_ur = sp.intersect1d(sp.where(self.x_idx >= (len(self.signal) - self.px_per_step - 2))[0],
                                          sp.where(self.y_idx >= (len(self.signal[0]) - self.px_per_step - 2))[0])
                
            for ind in wall_risk_ur:
                self.wall_hits[self.i, ind] = 1.0
                self.hit_wall[ind] = 1.0
                self.theta[self.i, ind] = self.ranfs[self.ranf_count] * 90.0 + 180.0
                self.ranf_count = self.ranf_count + 1
                
            wall_risk_left = sp.setdiff1d(sp.setdiff1d(sp.where(self.x_idx <= self.px_per_step)[0],
                                                       wall_risk_ul),
                                          wall_risk_ll)
                
            for ind in wall_risk_left:
                self.wall_hits[self.i, ind] = 1.0
                self.hit_wall[ind] = 1.0
                self.theta[self.i, ind] = (self.ranfs[self.ranf_count] * 180.0 + 270.0) % 360.0
                self.ranf_count = self.ranf_count + 1
                
            wall_risk_r = sp.setdiff1d(sp.setdiff1d(sp.where(self.x_idx >=
                                                             (len(self.signal) - self.px_per_step - 2))[0],
                                                    wall_risk_ur),
                                       wall_risk_lr)
            
            for ind in wall_risk_r:
                self.wall_hits[self.i, ind] = 1.0
                self.hit_wall[ind] = 1.0
                self.theta[self.i, ind] = self.ranfs[self.ranf_count] * 180.0 + 90.
                self.ranf_count = self.ranf_count + 1
                
            wall_risk_low = sp.setdiff1d(sp.setdiff1d(sp.where(self.y_idx <= self.px_per_step)[0],
                                                      wall_risk_lr),
                                         wall_risk_ll)
                
            for ind in wall_risk_low:
                self.wall_hits[self.i, ind] = 1.0
                self.hit_wall[ind] = 1.0
                self.theta[self.i, ind] = self.ranfs[self.ranf_count] * 180.0
                self.ranf_count = self.ranf_count + 1
                
            wall_risk_u = sp.setdiff1d(sp.setdiff1d(sp.where(self.y_idx >= (len(self.signal[0])-
                                                                            self.px_per_step - 2))[0],
                                                    wall_risk_ur),
                                       wall_risk_ul)
            for ind in wall_risk_u:
                self.wall_hits[self.i, ind] = 1.0
                self.hit_wall[ind] = 1.0
                self.theta[self.i, ind] = self.ranfs[self.ranf_count] * 180.0 + 180.0
                self.ranf_count = self.ranf_count + 1

        # wall behavior is stop
        elif(self.wall_behavior=='stop'):
            #print("Wall behavior is stop.")
            wall_risk = sp.intersect1d(sp.intersect1d(sp.where(self.x_idx <= self.px_per_step)[0],
                                                      sp.where(self.y_idx <= self.px_per_step)[0]),
                                       sp.intersect1d(sp.where(self.x_idx >=
                                                               len(self.signal) - self.px_per_step-2)[0],
                                                      sp.where(self.y_idx >=
                                                               len(self.signal[0]) - self.px_per_step-2)[0]))
            for ind in wall_risk:
                self.wall_hits[self.i, ind] = 1.0
                self.hit_wall[ind] = 1.0
                self.is_walking[ind] = 0.0


        # if walking, determine probability of stop and turn in next interval
        # and determine if fly stops or turns.
        
        walking_flies = sp.where(self.is_walking)[0]
        stopped_flies = sp.where(sp.logical_not(self.is_walking))[0]
        
        stopping_flies = sp.intersect1d(walking_flies,
                                        sp.where(self._poisson_happen(self.ws_fn[self.i]))[0])
        turning_flies = sp.intersect1d(walking_flies,
                                       sp.where(self._poisson_happen(sp.full(self.num_flies,
                                                                             self.turn_lambda)))[0])
        # stopping flies can't turn
        turning_flies = sp.setdiff1d(turning_flies, stopping_flies)
        keep_walking_flies = sp.setdiff1d(sp.setdiff1d(walking_flies, stopping_flies), turning_flies)

        # if stopped, determine the probability of walking
        start_walking_flies = sp.intersect1d(stopped_flies, sp.where(
            self._poisson_happen(self.sw_fn[self.i]))[0])

        keep_stopped_flies = sp.setdiff1d(stopped_flies, start_walking_flies)
        
        # if flies must stop when they get to a wall, add wall flies to stopped flies
        if(self.wall_behavior == 'stop'):
            keep_stopped_flies = sp.intersect1d(keep_stopped_flies,
                                                sp.where(self.hit_wall))


        # determine which flies will turn left and right
        turnL_idxs = sp.intersect1d(turning_flies,
                                    sp.where(self._happen(self._turnL_prob()))[0])
        turnR_idxs = sp.setdiff1d(turning_flies, turnL_idxs)

        # left turns

        del_theta = (self.turn_mean + self.turn_std * 
                     self.randns[self.randn_count : self.randn_count + len(turnL_idxs)])

        self.randn_count = self.randn_count + len(turnL_idxs)
        self.del_theta[self.i, turnL_idxs] = del_theta

        # right turns
        del_theta = (-self.turn_mean + self.turn_std *
                     self.randns[self.randn_count : self.randn_count + len(turnR_idxs)])
        self.randn_count = self.randn_count + len(turnR_idxs)
        self.del_theta[self.i, turnR_idxs] = del_theta

        #
        # updating instance variables
        #

        # walking flies that will stop
        self.is_walking[stopping_flies] = False
        self.walks[self.i, stopping_flies] = 0.0
        self.stops[self.i, stopping_flies] = 1.0
        self.turns[self.i, stopping_flies] = 0.0
        self.last_switch[stopping_flies] = self.i
        self.del_theta[self.i, stopping_flies] = 0.0
        
        # turning flies
        self.walks[self.i, turning_flies] = 1.0
        self.stops[self.i, turning_flies] = 0.0
        self.turns[self.i, turning_flies] = 1.0

        # walking flies that won't turn or stop
        del_theta = (self.no_turn_mean + self.no_turn_std *
                     self.randns[self.randn_count : self.randn_count + len(keep_walking_flies)])
        self.randn_count = self.randn_count + len(keep_walking_flies)
        self.del_theta[self.i, keep_walking_flies] = del_theta
        
        self.walks[self.i, keep_walking_flies] = 1.0
        self.stops[self.i, keep_walking_flies] = 0.0
        self.turns[self.i, keep_walking_flies] = 0.0

        # stopped flies
        self.turns[self.i, stopped_flies] = 0.0
        self.del_theta[self.i, stopped_flies] = 0.0
        
        # stopped flies that will walk
        self.is_walking[start_walking_flies] = 1.0
        self.walks[self.i, start_walking_flies] = 1.0
        self.stops[self.i, start_walking_flies] = 0.0
        self.last_switch[start_walking_flies] = self.i

        # stopped flies that will stay stopped
        self.walks[self.i, keep_stopped_flies] = 0.0
        self.stops[self.i, keep_stopped_flies] = 1.0
                    
        # update time, position coords

        # update velocities
        self.vx[self.i, walking_flies] = sp.cos(self.theta[self.i,walking_flies] * sp.pi / 180.) * self.speed
        self.vx[self.i, stopped_flies] = 0.0               
        self.vy[self.i, walking_flies] = sp.sin(self.theta[self.i,walking_flies] * sp.pi / 180.) * self.speed
        self.vy[self.i, stopped_flies] = 0.0

        if(self.i < self.num_steps-1):
            self.time[self.i+1] = self.time[self.i] + self.delta_t
            # update positions
            self.x[self.i+1] = self.x[self.i] + self.vx[self.i] * self.delta_t
            self.x_idx = sp.array(sp.rint(self.x[self.i+1] / self.delta_pos), dtype=int) 

            self.y[self.i+1] = self.y[self.i] + self.vy[self.i] * self.delta_t
            self.y_idx = sp.array(sp.rint(self.y[self.i+1] / self.delta_pos), dtype=int)

            self.theta[self.i+1] = self.theta[self.i] + self.del_theta[self.i]
            
    # the walk-stop rate, which follows last-whiff model
    def _ws_rate(self):
        # if the fly hasn't encountered any whiffs, it won't have
        # any modification to its base rate
        R = sp.zeros(self.num_flies)

        whiff_fly_idxs = sp.where(self.last_whiff >= 0)[0]
        
        R[whiff_fly_idxs] = sp.exp(-1*(self.time[self.i, whiff_fly_idxs] -
                                       self.last_whiff[whiff_fly_idxs])/self.ws_tau_R)
        
        res = self.ws_lambda_0 + self.ws_del_lambda * R
        
        return res
    
    # the stop-walk rate, which follows history dependence model
    def _sw_rate(self):
        conv = sp.zeros(self.num_flies)
        for fly in range(self.num_flies):
            switch_idx = self.last_switch[fly]
            hits_past_switch = self.whiff_hits[switch_idx:self.i+1,fly]
            len_stop = len(hits_past_switch)
            # exponential decay filter
            decay_fn = sp.exp(
                sp.negative((self.time[switch_idx:self.i+1, fly]-
                             sp.full_like(self.time[switch_idx:self.i+1,fly],self.time[switch_idx, fly])))/ 
                sp.full_like(self.time[switch_idx:self.i+1, fly],self.sw_tau_H))

            conv[fly] = (sp.convolve(decay_fn, 
                                         hits_past_switch)
                         [-min(len(decay_fn), len(hits_past_switch))])

        res = self.sw_lambda_0 + self.sw_del_lambda * conv
        return res
    
    def _turnL_prob(self):
        one_idxs = sp.where(self.theta[self.i] <= 180)[0]
        neg_one_idxs = sp.where(self.theta[self.i] > 180)[0]
        
        turn_dir = sp.zeros(self.num_flies)
        base_turnL_rate = sp.zeros(self.num_flies)

        turn_dir[one_idxs] = 1.
        base_turnL_rate[one_idxs] = self.base_upturn_rate
        
        turn_dir[neg_one_idxs] = -1.
        base_turnL_rate[neg_one_idxs] = 1 - self.base_upturn_rate

        return sp.maximum(sp.minimum((1/(1+sp.exp(-turn_dir*self.turn_alpha * self.wt[self.i]))),
                                     sp.full(self.num_flies,1)),
                                  sp.full(self.num_flies,0))

    
    def _wt(self):
        # exponential decay filter
        len_int = len(self.hits_window)
        start_window_idx = self.i - len_int + 1
        decay_fn = sp.exp(
            sp.negative((self.time[start_window_idx:self.i+1]-sp.full_like(self.time[start_window_idx:self.i+1],
                                                                   self.time[start_window_idx]))/ 
                        sp.full_like(self.time[start_window_idx:self.i+1],self.tau_turn)))

        w = sp.zeros(self.num_flies)
    
        for fly in range(self.num_flies):
            w[fly] = sp.convolve(decay_fn[:,fly], 
                                 self.hits_window[:,fly])[-min(len(decay_fn), len(self.hits_window))]
    
        self.wt[self.i] = w
        
    def _happen(self, prob):
        rand = self.ranfs[self.ranf_count : self.ranf_count + self.num_flies]
        self.ranf_count = self.ranf_count + self.num_flies
        return (rand < prob)
    
    # rate is an array of length self.num_flies
    def _poisson_happen(self, rate):
        prob = rate*self.delta_t
       
        return(self._happen(prob))
    


    # calculates an elliptical antenna for the fly
    #
    # returns the indices in this ellipse as a tuple (first entry = x vals, second = y vals)
    def make_antenna(self,fly_x_idx, fly_y_idx, fly_theta, r_from_fly_center=8.05, a=5.58, b=1.49):
        #
        # determine coordinates of center of ellipse
        #
        
        # coordinates relative to fly center
        x_loc_from_fly = sp.rint(r_from_fly_center * sp.cos(fly_theta * sp.pi / 180.))
        y_loc_from_fly = sp.rint(r_from_fly_center * sp.sin(fly_theta * sp.pi / 180.))
        
        # coordinates relative to origin (-20,-20) from fly center
        center_x = x_loc_from_fly + 20
        center_y = y_loc_from_fly + 20
                   
        # get x,y pixel grid
        x, y = sp.ogrid[:40,:40]
        
        #
        # determine ellipse equation and mask
        #
        
        A = fly_theta * sp.pi / 180. + sp.pi / 2  # ellipse angle in radians
        ellipse_val_1 = sp.square(((x - center_x) * sp.cos(A) + 
                                   (y - center_y) * sp.sin(A)) / a)
        ellipse_val_2 = sp.square(((x - center_x) * sp.sin(A) -
                                   (y - center_y) * sp.cos(A)) / b)
        
        ellipse_mask = (ellipse_val_1 + ellipse_val_2) <= 1.
        ellipse_idxs = sp.where(ellipse_mask)
        ellipse_idxs = (ellipse_idxs[0] + fly_x_idx-20, ellipse_idxs[1] + fly_y_idx-20)
        return ellipse_idxs

######## Error for when arguments are not correct. #########
class ArgError(Exception):
    def __init__(self, message):
        self.message = message
