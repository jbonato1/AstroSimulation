import numpy as np
from joblib import Parallel, delayed
import pandas as pd


def histedges_equalN(x, nbin):
    '''
    Function to compute the Uniform (equally populated) histogram edges for *nbin* using the observation array "x".
    Returns: np.array with *nbins*+1 edges.
    '''
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))

def convert_sp_bin(arr,nbin):
    new = arr.copy()
    for i in range(1,len(nbin)):
        if i==1:
            pt = np.where((arr>=nbin[i-1])& (arr<=nbin[i]))
        else: 
            pt = np.where((arr>nbin[i-1])& (arr<=nbin[i]))
        #print(pt)
        if pt[0].shape[0]>0:
            new[pt[0]]=i-1
    return new 

def intensity_bin(arr,nbin):
    new = arr.copy()
    if new.max()==0:
        return new
    else:
#         new[new>0]=1
#         new[new<1]=0
        new/=new.max()
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


        
def gen_RMAT_single(mat,space,n_spatial_bin=12,int_bin=2):
    
    first_cycle = True
    verbose=True

    data_couple=[]
    counter = 0
    id_couple = 0
    S=[]
    for i in range(mat.shape[0]):

        qq = histedges_equalN(space, nbin=n_spatial_bin)
        qq = np.insert(qq,0,0)
        S.append(convert_sp_bin(space,qq))
        
        v1 = intensity_bin(mat[i,:],int_bin)
        data_couple.append(v1[:,np.newaxis])

    return S,data_couple
    
def gen_RMAT_single_exp(list_,n_spatial_bin=12,int_bin=2):
    first_cycle = True
    verbose=True
    N = len(list_)
   
    
    name_id_couple = []
    data_couple=[]
    counter = 0
    id_couple = 0
    S=[]
    for name in list_:
        #print(name)
        
        df = pd.read_csv(name)  
        qq = histedges_equalN(df['Positive_Positions'], nbin=n_spatial_bin)
        S.append(convert_sp_bin(df['Positive_Positions'].values,qq))

        v1 = intensity_bin(df['Positive_Intensities'].values,int_bin)
        data_couple.append(v1[:,np.newaxis])

    
    return S,data_couple




def binarize(arr,st_pt,end_pt,nbin,new):
    delta = (end_pt-st_pt)/nbin
    
    
    if st_pt<60:
        off = 0
    elif st_pt<120 and st_pt>=60:
        off = 1*nbin
    else:
        off = 2*nbin
        
    for i in range(nbin):
        
        if i<(nbin-1):
            
            #print(i, delta*i+st_pt, delta*(i+1)+st_pt)
            pt = np.where((arr>=(delta*i+st_pt))& (arr<(delta*(i+1)+st_pt)))
        
        else:
            
            #print(i, delta*i+st_pt, end_pt)
            pt = np.where((arr>=(delta*i+st_pt))& (arr<=end_pt))
        
        if pt[0].shape[0]>0: 
            
            new[pt[0]]=i+off
    return new

def bin_space_visCues(arr,min_in=0,max_in=171.11011490962943,nbin = 3):#171.11011490962943
    
    new = arr.copy()
    #first cue start-60
    new = binarize(arr,min_in,60,nbin,new)
    #second cue 60-120
    new = binarize(arr,60,120,nbin,new)
    #third cue 120-180
    new = binarize(arr,120,max_in,nbin,new)
    
    return new 

class shuffling_VC_constr():
    def __init__(self,case):
        #np.random.seed(42)
        self.case = case
          

    def split(self,X,y,groups=None):
        Xnew = np.empty((0,0),dtype=np.float64)
        Tr = X.shape[0]
        
        M = y.max()+1
        #print('CIAO',M,X.shape,y.shape)
#         input('qqqq')

        for i in [2,3,4,5,6,7,8,10,20,30]:
            if M/i==3:
                #print(i)
                edges = [i*j for j in range(4)]
            #else:
                #print('wrong')
        
        #edges = [0,3,6,9]
        #print(edges,np.unique(y,return_counts=True))
        #input('Test')
        for i in range(1,len(edges)):
            
            pt = np.where((y>=edges[i-1])&(y<edges[i]))
            #print(np.unique(y[pt[0]]))
            if len(X.shape)>1:
                new = X[pt[0],:].copy()
            else:
                new = X[pt[0]].copy()
                
            new_y = y[pt[0]].copy()
            
            if i==self.case:
                np.random.shuffle(new_y)
                
            if self.case==-1:
                np.random.shuffle(new_y)
            
            if Xnew.size>0:
                Xnew = np.concatenate((Xnew,new),axis=0)
                y_label = np.concatenate((y_label,new_y),axis=0)
            else:
                Xnew = new
                y_label = new_y
               
        return Xnew,y_label


def gen_RMAT_single_eqSpace(list_,n_spatial_bin=12,int_bin=2,case=-1,iterVS=100):

    data_couple=[]    
    S=[]
    shuf_cnst = shuffling_VC_constr(case=case)
    cnt=0
    for name in list_:
        
        
        df = pd.read_csv(name) 
        space_original = bin_space_visCues(df['Positive_Positions'].values,nbin=int(n_spatial_bin/3))

        v1 = intensity_bin(df['Positive_Intensities'].values,int_bin)
        
        space = np.empty((space_original.shape[0],iterVS+1))
        inten = np.empty((space_original.shape[0],iterVS+1))
        space[:,0] = space_original
        inten[:,0] = v1
        if cnt==0:
            print(np.unique(space_original,return_counts=True))
        for i in range(iterVS):
            inten[:,i+1],space[:,i+1] = shuf_cnst.split(v1,space_original)# 
            
        S.append(space)
        data_couple.append(inten)
        cnt+=1
    return S,data_couple


def gen_RMAT_single_VS(mat,space,n_spatial_bin=12,int_bin=2,case=-1,iterVS=100):

    data_couple=[]    
    S=[]
    shuf_cnst = shuffling_VC_constr(case=case)
    
#     print(mat.shape,space.shape,n_spatial_bin,int_bin)
#     print(np.unique(space))
#     input('')
    
    for j in range(mat.shape[0]):

        qq = histedges_equalN(space, nbin=n_spatial_bin)
        qq = np.insert(qq,0,0)

        space_original = convert_sp_bin(space,qq)

        v1 = intensity_bin(mat[j,:],int_bin)
        
        spaceV = np.empty((space_original.shape[0],iterVS+1))
        inten = np.empty((space_original.shape[0],iterVS+1))
        spaceV[:,0] = space_original
        inten[:,0] = v1

        for i in range(iterVS):
            inten[:,i+1],spaceV[:,i+1] = shuf_cnst.split(v1,space_original)# 
            
        S.append(spaceV)
        data_couple.append(inten)
    
    return S,data_couple


