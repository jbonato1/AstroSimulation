import glob
import os
import pandas as pd
import numpy as np

import pickle
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

#import matplotlib
#matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import matlab.engine

# !pip install tqdm
from tqdm import tqdm
from MI_modules import *
from joblib import Parallel, delayed
import time

### MI function
def mod_MI_single(data,S,i,blocks):
    path='/home/jbonato/Documents/infotoolbox_v1.1.0b4_eugenio/'
    eng = matlab.engine.start_matlab()
    flag=False
    while(not(flag)):
        try:
            eng.addpath (path, nargout= 0 )
            eng.startup_infotoolbox(nargout=0)
            flag=True
        except:
            flag=False
            
    
    iter_boot = 100
    iter_qe = 0
    out_res =[]
    for j in range(blocks[i],blocks[i+1]):
        
        data_rMat = matlab.double(data[j].T.tolist()) 
        data_S = matlab.double(S[i].tolist()) 

        qq,ww = eng.buildr(data_S,data_rMat,nargout=2)
        
        structure = eng.gen_struct(ww,'dr','naive',float(iter_boot),float(0))
        varargout = eng.information(qq, structure, 'I',nargout=1)        
        
        structure = eng.gen_struct(ww,'dr','pt',float(iter_boot),float(0))
        varargout1 = eng.information(qq, structure, 'I',nargout=1) 
        
        structure = eng.gen_struct(ww,'dr','qe',float(iter_boot),float(1))
        varargout2 = eng.information(qq, structure, 'I',nargout=1) 
        
        out_res.append([j,np.concatenate((np.asarray(varargout),np.asarray(varargout1),np.asarray(varargout2)),axis=0)])
    eng.quit()

    return out_res

def computeMI_par_single(mat,space,spatial_bin,int_bin):
    
    list_cell = np.arange(mat.shape[0])
    
    res_dict = {}
    S,data = gen_RMAT_single(mat,space,n_spatial_bin=spatial_bin,int_bin=int_bin)
    
    blocks  = [i*2 for i in range(mat.shape[0]//2)]
    blocks.append(mat.shape[0])
    print(blocks)
    results_list = Parallel(n_jobs=23,require='sharedmem',verbose=1)(delayed(mod_MI_single)(data,S,i,blocks) for i in range(len(blocks)-1))
    cnt=0
    for jj in range(len(results_list)):
        for results in results_list[jj]:
            res_dict['roi_'+str(list_cell[results[0]])+'_'+str(spatial_bin)+'_'+str(int_bin)] = results[1][:,:,np.newaxis]
            if results[1][2,0] >= np.percentile(results[1][2,1:],95):
                cnt+=1
    print('RES',100*(cnt/20))
    ###clean matlab temp files
    path='/home/jbonato/Documents/infotoolbox_v1.1.0b4_eugenio/'
    eng = matlab.engine.start_matlab()
    flag=False
    while(not(flag)):
        try:
            eng.addpath (path, nargout= 0 )
            flag=True
        except:
            flag=False
    eng.reh_cmd(nargout=0)
    eng.quit()
    print('Cache cleaned')


#     file2 = open(folder+'.pkl','wb')
#     pickle.dump(res_dict,file2)
#     file2.close()

    print('END',res_dict.keys())
    return res_dict



def computeMI_par_single_exp(list_cell,folder,spatial_bin,int_bin):
    
    
    print("Astro int bin",int_bin,'SPace bins: ',spatial_bin)
    res_dict = {}
    
    S,data = gen_RMAT_single_exp(list_cell,n_spatial_bin=spatial_bin,int_bin=int_bin)
    
    blocks  = [i*10 for i in range(len(list_cell)//10)]
    blocks.append(len(list_cell))
    print(blocks)
    results_list = Parallel(n_jobs=23,require='sharedmem',verbose=1)(delayed(mod_MI_single)(data,S,i,blocks) for i in range(len(blocks)-1))

    for jj in range(len(results_list)):
        for results in results_list[jj]:
            name = os.path.basename(list_cell[results[0]])
            res_dict['roi_'+name[:-4]+'_'+str(spatial_bin)+'_'+str(int_bin)] = results[1][:,:,np.newaxis]

    ###clean matlab temp files
    path='/home/jbonato/Documents/infotoolbox_v1.1.0b4_eugenio/'
    eng = matlab.engine.start_matlab()
    flag=False
    while(not(flag)):
        try:
            eng.addpath (path, nargout= 0 )
            flag=True
        except:
            flag=False
    eng.reh_cmd(nargout=0)
    eng.quit()
    print('Cache cleaned')
    file2 = open(folder+'_S'+str(spatial_bin)+'_R'+str(int_bin)+'.pkl','wb')
    pickle.dump(res_dict,file2)
    file2.close()


### MI function
def mod_MI_single_plug(data,S,i,blocks):
    path='/home/jbonato/Documents/infotoolbox_v1.1.0b4_eugenio/'
    eng = matlab.engine.start_matlab()
    flag=False
    while(not(flag)):
        try:
            eng.addpath (path, nargout= 0 )
            eng.startup_infotoolbox(nargout=0)
            flag=True
        except:
            flag=False
            
    
    iter_boot = 100
    iter_qe = 0
    out_res =[]
    for j in range(blocks[i],blocks[i+1]):
        
        data_rMat = matlab.double(data[j].T.tolist()) 
        data_S = matlab.double(S[i].tolist()) 

        qq,ww = eng.buildr(data_S,data_rMat,nargout=2)
        structure = eng.gen_struct(ww,'dr','naive',float(iter_boot),float(0))
        varargout = eng.information(qq, structure, 'I',nargout=1)        
        
#         structure = eng.gen_struct(ww,'dr','pt',float(iter_boot),float(0))
#         varargout1 = eng.information(qq, structure, 'I',nargout=1) 
        
#         structure = eng.gen_struct(ww,'dr','qe',float(iter_boot),float(1))
#         varargout2 = eng.information(qq, structure, 'I',nargout=1) 
        
        out_res.append([j,np.concatenate((np.asarray(varargout),np.asarray(varargout),np.asarray(varargout)),axis=0)])
    eng.quit()

    return out_res

def computeMI_par_single_plug(mat,space,spatial_bin,int_bin):
    
    list_cell = np.arange(mat.shape[0])
    
    res_dict = {}
    S,data = gen_RMAT_single(mat,space,n_spatial_bin=spatial_bin,int_bin=int_bin)
    blocks  = [i*2 for i in range(mat.shape[0]//2)]
    blocks.append(mat.shape[0])
    print(blocks)
    results_list = Parallel(n_jobs=23,require='sharedmem',verbose=1)(delayed(mod_MI_single_plug)(data,S,i,blocks) for i in range(len(blocks)-1))

    for jj in range(len(results_list)):
        for results in results_list[jj]:
            res_dict['roi_'+str(list_cell[results[0]])+'_'+str(spatial_bin)+'_'+str(int_bin)] = results[1][:,:,np.newaxis]

    ###clean matlab temp files
    path='/home/jbonato/Documents/infotoolbox_v1.1.0b4_eugenio/'
    eng = matlab.engine.start_matlab()
    flag=False
    while(not(flag)):
        try:
            eng.addpath (path, nargout= 0 )
            flag=True
        except:
            flag=False
    eng.reh_cmd(nargout=0)
    eng.quit()
    print('Cache cleaned')


#     file2 = open(folder+'.pkl','wb')
#     pickle.dump(res_dict,file2)
#     file2.close()

    print('END',res_dict.keys())
    return res_dict


################ Visual cue

def mod_MI_singleVS(couple_data,S,i,blocks):
    path='/home/jbonato/Documents/infotoolbox_v1.1.0b4_eugenio/'
    eng = matlab.engine.start_matlab()
    flag=False
    while(not(flag)):
        try:
            eng.addpath (path, nargout= 0 )
            eng.startup_infotoolbox(nargout=0)
            flag=True
        except:
            flag=False
    
    
    out_res =[]
    #print(couple_data[0].T.shape,S[0].T.shape)
    for j in range(blocks[i],blocks[i+1]):
        
        data_rMat = matlab.double(couple_data[j].T.tolist()) 
        data_S = matlab.double(S[j].T.tolist()) 
        varargout = eng.VisC_shuff(data_S,data_rMat,nargout=1)
        res_sign = np.asarray(varargout)
        out_res.append([j,res_sign])
    
    eng.quit()
    #print(out_res[0][1].shape)
    return out_res



def computeMI_par_single_exp_VC(list_cell,folder,spatial_bin,int_bin):
    
    
    print("Astro int bin",int_bin,'SPace bins: ',spatial_bin)
    res_dict = {}
    #S,couple_data = gen_RMAT_single_eqSpace(list_cell,n_spatial_bin=spatial_bin,int_bin=astro_int_bin,neuro=neuro,case=case
    S,data = gen_RMAT_single_eqSpace(list_cell,n_spatial_bin=spatial_bin,int_bin=int_bin)
    print(np.unique(S[0],return_counts=True))
    blocks  = [i*10 for i in range(len(list_cell)//10)]
    blocks.append(len(list_cell))
    print(blocks)
    results_list = Parallel(n_jobs=23,require='sharedmem',verbose=1)(delayed(mod_MI_singleVS)(data,S,i,blocks) for i in range(len(blocks)-1))
    
    
    for jj in range(len(results_list)):
        for results in results_list[jj]:
            name = os.path.basename(list_cell[results[0]])
            res_dict['roi_'+name[:-4]+'_'+str(spatial_bin)+'_'+str(int_bin)] = results[1][:,:,np.newaxis]

    ###clean matlab temp files
    path='/home/jbonato/Documents/infotoolbox_v1.1.0b4_eugenio/'
    eng = matlab.engine.start_matlab()
    flag=False
    while(not(flag)):
        try:
            eng.addpath (path, nargout= 0 )
            flag=True
        except:
            flag=False
    eng.reh_cmd(nargout=0)
    eng.quit()
    print('Cache cleaned')
    
    file2 = open(folder+'_S'+str(spatial_bin)+'_R'+str(int_bin)+'.pkl','wb')
    pickle.dump(res_dict,file2)
    file2.close()


    
def computeMI_par_single_VC(mat,space,spatial_bin,int_bin):
    
    list_cell = np.arange(mat.shape[0])
    res_dict = {}

    S,data = gen_RMAT_single_VS(mat,space,n_spatial_bin=spatial_bin,int_bin=int_bin)
    
    blocks  = [i*2 for i in range(len(list_cell)//2)]
    blocks.append(len(list_cell))
    print(blocks)
    results_list = Parallel(n_jobs=23,require='sharedmem',verbose=1)(delayed(mod_MI_singleVS)(data,S,i,blocks) for i in range(2))#len(blocks)-1
    
    cnt = 0 
    for jj in range(len(results_list)):
        for results in results_list[jj]:            
            res_dict['roi_'+str(list_cell[results[0]])+'_'+str(spatial_bin)+'_'+str(int_bin)] = results[1][:,:,np.newaxis]
            print(results[1][0,0],np.percentile(results[1][1:,0],95))
            if results[1][0,0] >= np.percentile(results[1][1:,0],95):
                cnt+=1
    print('RES',100*(cnt/20))
    ###clean matlab temp files
    path='/home/jbonato/Documents/infotoolbox_v1.1.0b4_eugenio/'
    eng = matlab.engine.start_matlab()
    flag=False
    while(not(flag)):
        try:
            eng.addpath (path, nargout= 0 )
            flag=True
        except:
            flag=False
    eng.reh_cmd(nargout=0)
    eng.quit()
    print('Cache cleaned')
    
    print('END',res_dict.keys())
    return res_dict

    
def shannonEntropy(probability):
    '''
    Function to compute the Shannon Entropy (H) in bits of the input *probability* distribution.
    If the probability distribitution isn't a PDF (i.e. sum(probability)!=1) it first converts the frequency distribution in a suitable PDF.
    Note: in order to avoid failure for 0 count frequencies (and PDF) in uses just NON-ZERO values.
    Returns: H
    '''
    if np.sum(probability) != 1:
        probability = probability/np.sum(probability)
    #print(probability, type(probability))    
    probability = probability[np.nonzero(probability)]
    #print(probability, type(probability))    
    H = -np.sum(probability*np.log2(probability))
    return(H)

##########
#test
import ctypes
import numpy as np 
import pandas as pd
from numpy.ctypeslib import ndpointer
panz_trev_bay = ctypes.cdll.LoadLibrary("./info_mod/panz_trev_96.so")
corr_bay = panz_trev_bay.panzeri_treves_96
corr_bay.restype = None
corr_bay.argtypes = [ndpointer(ctypes.c_double,flags="C_CONTIGUOUS"),
                   ctypes.c_int,
                   ctypes.c_double,
                   ndpointer(ctypes.c_double,flags="C_CONTIGUOUS")
                   ]


def mutual_Info(x, y, bins_x, bins_y):
    '''
    Function to compute the Mutual information of joint probability distributions.
    It uses shannonEntropy function to compute 'h_x','h_y','h_xy'
    mutual information is computed as_
    mi = 'h_x'+'h_y'-'h_xy'
    Input:
    x, y =  observations array 
    bins_x, bins_y = bins, either edges or bin number
    roi_id = Optional ROI identifier 
    
    Returns: df with 'h_x','h_y','h_xy','mi' + optional 'roi_id' columns and 1 row
    
    '''
    p_x = np.histogram(x, bins = 4)
    h_x = shannonEntropy(p_x[0])
    
    p_y = np.histogram(y, bins = bins_y)
    
    h_y = shannonEntropy(p_y[0])
    
    p_joint_xy = np.histogram2d(x, y, bins = [bins_x ,bins_y])
    h_xy = shannonEntropy(p_joint_xy[0])
    
    mi =  h_x + h_y - h_xy
    
    ### correction
#     N = np.sum(p_joint_xy[0])
#     H = p_joint_xy[0].T#/np.sum(p_joint_xy[0])
#     H = np.ascontiguousarray(H,dtype=np.double)
#     prob_Int=np.sum(H,axis=1).astype(np.double)
    
#     R_tilda_s = np.zeros((1,),dtype = np.double)
#     R_tilda = np.zeros((1,),dtype = np.double)
#     buff = np.zeros((1,),dtype=np.double)
#     N_I = np.empty((1,),dtype = np.intc)
#     N_I[0] = H.shape[1]
    
#     for j in range(H.shape[0]):
#         corr_bay( H[j,:],N_I[0],prob_Int[j], buff)
#         R_tilda_s+=buff*(prob_Int[j]/N)
#     corr_bay(np.sum(H,axis=0), N_I[0], N.astype(np.double),R_tilda)
#     bias = R_tilda-R_tilda_s
    return mi

def compMI_PAR(data,S,i,blocks):
    out_res=[]
    
    for j in range(blocks[i],blocks[i+1]):
        
        out = np.empty((101,1,1))
        data_comp = data[j]
        S_comp = S[j]
        #print('1111',data_comp.shape)
        for ii in range(101):#101
            out[ii,0,0] = mutual_Info(data_comp[:,ii], S_comp[:,ii], 4, 12)
            
        out_res.append([j,out])
    
    return out_res
    


def computeMI_par_single_VC2(mat,space,spatial_bin,int_bin):
    
    list_cell = np.arange(mat.shape[0])
    res_dict = {}

    S,data = gen_RMAT_single_VS(mat,space,n_spatial_bin=spatial_bin,int_bin=int_bin)#
    print(np.unique(data[0]))
    blocks  = [i for i in range(len(list_cell))]
    blocks.append(len(list_cell))
    print(blocks)
    results_list = Parallel(n_jobs=23,require='sharedmem',verbose=1)(delayed(compMI_PAR)(data,S,i,blocks) for i in range(2))#len(blocks)-1)
    
    cnt = 0 
    for jj in range(len(results_list)):
        for results in results_list[jj]:            
            res_dict['roi_'+str(list_cell[results[0]])+'_'+str(spatial_bin)+'_'+str(int_bin)] = results[1][:,:,np.newaxis]
            print(results[1][0,0,0],np.percentile(results[1][1:,0,0],95))
            if results[1][0,0,0] >= np.percentile(results[1][1:,0,0],95):
                cnt+=1
    print('RES',100*(cnt/20))
    print('END',res_dict.keys())
    return res_dict