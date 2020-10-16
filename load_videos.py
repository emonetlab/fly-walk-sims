"""
Functions and methods for loading data

Created by Nirag Kadakia at 10:34 11-02-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import scipy.io as spio
import os.path
import h5py

def load_vid_by_frm(subdir, file, frame, bck_sub=True, bck_file = None):
    """
    Load a single frame of a flywalk movie from the h5 file. 

    Args:
        subdir: string of directory within mahmut_demir/analysis
        file: string of encounter dataset file (-.mat)
        frame: int; frame to load
        bck_sub: bool; if True, subtract background

    Returns:
        frm_data: (n, m) array; data for frame. 
    """

    data_dir = '%s' % (subdir)
    vid_file = '%s/%s' %  (subdir, file)
    #bck_file = '%s.mat' %  (file)

    assert os.path.isfile(vid_file) == True, "%s not found" % vid_file
    with h5py.File(vid_file, 'r') as f:
        data_set = f['frames']
        frm_data = data_set[frame]

    if bck_sub == True:
        bck_file = '%s/%s' %(subdir, bck_file)
        assert os.path.isfile(bck_file) == True, "%s not found" % bck_file
        with h5py.File(bck_file, 'r') as f:
            bck_data = f['p']['bkg_img'][:]
            mask = f['p']['mask'][:]
            zero_idxs = sp.where(frm_data < bck_data)
            frm_data -= bck_data
            frm_data[zero_idxs] = 0.0
            frm_data = frm_data * mask

    return frm_data

