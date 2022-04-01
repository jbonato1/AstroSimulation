import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import pickle
from computeMI import computeMI_par_single

file = open('/home/jbonato/Documents/AstroEncInfo/param_sim_ALL.pkl','rb')
dict_fit = pickle.load(file)
file.close()

#set place field param
sigmaPF = 56/2 #56 cm da Curreli et al.
APF = 0.44


def gauss(x,mu,sigma,A):
    return (A)*np.exp(-(x-mu)**2/(2*sigma**2))#1/(np.sqrt(2*np.pi)*sigma)

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
sim = 20
trial_num = [4]
ROInum = 20
Sbins = 12

#centerPF = 90
centerPF = np.random.uniform(1,180,ROInum)


dict_gen={}
for alpha in [1,0.5]:
    for trial_id in range(len(trial_num)):
        trial = 63
        for simulation in range(sim):
            
            stimS = np.arange(Sbins)
            stimScoord = stimS*(180/Sbins)+(180/Sbins)/2
            centerPF_bin = binarize_space(centerPF,Sbins)
            # for single pos PF
#            centerPF_bin = binarize_space(np.asarray([centerPF]),Sbins)[0]


            # retrieve the standard dev
            dict_std = dict_fit['Fit_space_'+str(Sbins)]
            f_s = np.zeros((Sbins))
            std_s = np.zeros((Sbins))
            
            for roi in range(ROInum):

                for i in range(Sbins):

                    pos = abs(stimS[i]-centerPF_bin[roi])


                    slope, intercept, r, p, se,mean,std_mean = dict_std['Pos_'+str(int(pos))]
                    f_s[roi,i] = mean
                    std_s[roi,i] = mean*slope+intercept
                if alpha==0:
                    buff = np.mean(std_s[roi,:])
                    std_s[roi,:] = buff
                f_s[roi,:] = f_s[roi,:] +(1-alpha)*(-f_s[roi,:]+np.mean(f_s[roi,:])) 

################# for single PF position                
#             for i in range(Sbins):
#                 pos = abs(stimS[i]-centerPF_bin)
                
                
#                 slope, intercept, r, p, se,mean,std_mean = dict_std['Pos_'+str(int(pos))]
#                 f_s[i] = mean
#                 std_s[i] = mean*slope+intercept
#             if alpha==0:
#                 buff = np.mean(std_s)
#                 std_s[:] = buff
#             f_s = f_s +(1-alpha)*(-f_s+np.mean(f_s)) 

            data = np.empty((ROInum,Sbins,trial))

            for roi in range(ROInum):
                for sp in range(Sbins):
                    val = std_s[sp] * np.random.randn(trial) + f_s[sp]
                    data[roi,sp,:] = val


            data[data<0]=0
            data = data.transpose(0,2,1).reshape(ROInum,trial*stimS.shape[0])
            spaceS = np.tile(stimScoord,trial)
            print('Data shape: ',data.shape,' in sim: ',simulation)
            ########### compute MI
            for R_bin in [2,3,4,5,8,10,20,40,80]:
                dict_out = computeMI_par_single(data,spaceS,12,R_bin)
                #print(dict_out['roi_0_12_4'][:,0,0],np.percentile(dict_out['roi_0_12_4'][:,1:,0],95,axis=1))
                if simulation==0:
                    if R_bin==2:
                        dict_gen = dict_out
                    else:
                        dict_gen = {**dict_gen, **dict_out}
                else:
                    for key in dict_out:
                        dict_gen[key] = np.concatenate((dict_gen[key],dict_out[key]),axis=2)
            #print(dict_gen.keys())


        file2 = open('/home/jbonato/Documents/AstroEncInfo/Results_all/MI_25tr_R_space12_alpha'+str(alpha)+'.pkl','wb')
        pickle.dump(dict_gen,file2)
        file2.close()