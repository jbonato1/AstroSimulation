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
        
        for astro_int_bin in [6,10,12]:
            for spatial_bin in [8,12,16,20,40,60,80]:
                print('BIN astro: ',astro_int_bin)
                
                computeMI_par_single_exp(list_a,'/home/jbonato/Documents/AstroEncInfo/EXP/Results/'+fov_n[-9:],spatial_bin,astro_int_bin)
