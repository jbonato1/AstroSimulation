import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import pickle
from computeMI import computeMI_par_single

#
file = open('/home/jbonato/Documents/AstroEncInfo/param_sim.pkl','rb')
dict_fit = pickle.load(file)
file.close()


#set place field param
centerPF = 90
sigmaPF = 56/2 #56 cm da Curreli et al.
APF = 0.44

def gauss(x,mu,sigma,A):
    return (A)*np.exp(-(x-mu)**2/(2*sigma**2))


def binarize_space(arr,nbin,min_in=0,max_in=180):
    new = arr.copy()
    new[new>max_in]=max_in
    new-=min_in#new.min()
    print(new.shape)
    new=new/max_in
    #print(new.max())
    delta = 1/nbin
    for i in range(nbin):
        #print(nbin-i-1)
        if delta*(nbin-i-1)!=0:
            pt = np.where((new>delta*(nbin-i-1) )& (new<=delta*(nbin-i)))
        else:
            pt = np.where((new>=delta*(nbin-i-1) )& (new<=delta*(nbin-i)))
        #print("INTerval",delta*(nbin-i-1),delta*(nbin-i))
        #print("BIN",nbin-i-1,"LEN",len(pt[0]))
        if pt[0].shape[0]>0:
            new[pt[0]]=nbin-i-1
    return new 

np.random.seed(42)
sim = 10
trial_num = [0,1,2,3,4,5,6]#,7,8,9,10]
ROInum = 20
alpha = 0.5
Sbins = 16


dict_gen={}
for alpha in [1,0.5,0]:
    for trial_id in range(len(trial_num)):
        trial = 2**(trial_num[trial_id])*4
        
        f_s = np.zeros((ROInum,Sbins))
        std_s = np.zeros((ROInum,Sbins))
        
        for simulation in range(sim):
            stimS = np.arange(Sbins)
            stimScoord = stimS*(180/Sbins)+(180/Sbins)/2
            centerPF_bin = binarize_space(np.asarray([centerPF]),Sbins)[0]

            #compute the f(s): mean along space modulated by a gaussian place field
            #f_s = gauss(stimScoord,centerPF,sigmaPF,APF)+(1-alpha)*(-gauss(stimScoord,centerPF,sigmaPF,APF)+np.mean(gauss(stimScoord,centerPF,sigmaPF,APF)))
            
            # retrieve the standard dev
            dict_std = dict_fit['Fit_space_'+str(Sbins)]


            for i in range(Sbins):
                pos = abs(stimS[i]-centerPF_bin)

                slope, intercept, r, p, se,mean,std_mean = dict_std['Pos_'+str(int(pos))]
                for kk in range(ROInum):
                    if simulation==0:
                        f_s[kk,i] = std_mean * np.random.randn(1) + mean
                    std_s[kk,i] = f_s[kk,i]*slope+intercept

            if alpha==0:
                buff = np.mean(std_s,axis=1)
                std_s[:,:] = buff[:,np.newaxis]
            f_s = f_s +(1-alpha)*(-f_s+np.mean(f_s,axis=1)[:,np.newaxis]) 
            print(f_s[0,:])
            data = np.empty((ROInum,Sbins,trial))

            for roi in range(ROInum):
                for sp in range(Sbins):
                    val = std_s[roi,sp] * np.random.randn(trial) + f_s[roi,sp]
                    data[roi,sp,:] = val


            data[data<0]=0
            data = data.transpose(0,2,1).reshape(ROInum,trial*stimS.shape[0])
            spaceS = np.tile(stimScoord,trial)
            print('Data shape: ',data.shape,' in sim: ',simulation)
            ########### compute MI
            dict_out = computeMI_par_single(data,spaceS,Sbins,4)
            if simulation==0:
                dict_gen = dict_out
            else:
                for key in dict_out:
                    dict_gen[key] = np.concatenate((dict_gen[key],dict_out[key]),axis=2)
        


        file2 = open('/home/jbonato/Documents/AstroEncInfo/Resultsv2/MI_trial'+str(trial)+'_R4_space12_alpha'+str(alpha)+'.pkl','wb')
        pickle.dump(dict_gen,file2)
        file2.close()