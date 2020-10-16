# fly-walk-sims

This repository contains scripts to run simulations of virtual agents navigating a complex odor plume.

## Requirements

The code requires Python 3 and has been tested on Python version 3.6.2. 

Plume video data must be supplied separately. The video are saved in .mat format, as described in the accompanying **fly-walk** repository (https://git.yale.edu/emonetlab/fly-walk). 

## Usage

To run a single simulation:

```
python complex_video_full_fly_run_script.py 0
```

Here, the command line argument '0' refers to the job number, which is used to set the random number seed for the simulation. The command line argument can be any integer.

This script instantiates an object of the **VectorFlies** class, which resides in the **complex_video_full_fly_simulator.py** module. The used-specified arguments for this class are described in detail in the VectorFlies docstring. The **VectorFlies.go()** method runs the actual simulation. This method loads the plume data from the appropriate .mat data, which is loaded frame-by-frame with h5py using the **load_vid_by_frm** function in **load_videos.py**.

A batch or array of simulations can be run in parallel (say on a computing cluster) by submitting **submit_complex_video_full_fly_simulation.sh** to the job manager.

## Authors

Hope D. Anderson (UChicago)

Viraaj Jayaram (Yale)