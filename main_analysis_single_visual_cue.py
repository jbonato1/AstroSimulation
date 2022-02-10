import glob
import os
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from joblib import Parallel, delayed

from computeMI import *
import sys


folder = '/home/jbonato/Documents/CausalInf_astro/SVM_dec_astro/data/wheel_obs_data/'
         
FOV_name = glob.glob(folder+"*")
print(FOV_name)


for fov_n in FOV_name:
        print("Analyze: ",fov_n[-9:])
        list_a = glob.glob(fov_n+'/*')
        
        for astro_int_bin in [2,4,8,20,80]:
            for spatial_bin in [60]:
                print('BIN astro: ',astro_int_bin)
                
                computeMI_par_single_exp_VC(list_a,'/home/jbonato/Documents/AstroEncInfo/EXP/Results_VC/'+fov_n[-9:],spatial_bin,astro_int_bin)
