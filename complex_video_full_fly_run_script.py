#This is a script to run the simulation encoded in complex_video_full_fly_simulator.py
#The initializations used for the simulations can be seen below. The job is read from the
#cluster submission script. 

import complex_video_full_fly_simulator as nav
import scipy as sp
import sys

flies = nav.VectorFlies((90,270),(308,309),(8,176),sp.arange(0,11690),
                        11690,100,save_data=True, speed = 10.1,
                        whiff_sigma_coeff = 2.749,
                        big_arena=True,bck_sub=1,wall_behavior='walk',signal_mean=-0.139,signal_std=1.08,
                        job=sys.argv[1], real_antenna=True)

flies.go()
