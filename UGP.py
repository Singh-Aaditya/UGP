## Create directory path
import os
dataset_path = '/home/aaditya/Bach10/'
song_paths = []
for song_name in os.listdir(dataset_path) :
    if (song_name != '.DS_Store') : song_paths.append(dataset_path + song_name + '/')
length = len(song_paths)
song_paths = sorted(song_paths)
print(song_paths)


## Create directories
mats_and_wavs = [[] for k in range(len(song_paths))]
for i in range(len(song_paths)):
    mats_and_wavs[i].append(song_paths[i]+song_paths[i][21:-1]+'-GTF0s.mat')
    mats_and_wavs[i].append(song_paths[i]+song_paths[i][21:-1]+'.wav')
print(mats_and_wavs[0][1])


## Read wavfile for the first song
import librosa
import IPython
song, sample_rate = librosa.load(mats_and_wavs[0][1])
# IPython.display.Audio(data = clipped_song, rate = sample_rate)
# sample_rate = 22050


## Compute STFT
window_size = int(0.084*sample_rate)
hop_size = int(0.01*sample_rate)
# Stft = librosa.stft(song, n_fft = window_size, hop_length=hop_size)
# Vft, phase = librosa.magphase(Stft)

# Plot
import numpy as np
#import matplotlib.pyplot as plt
#import librosa.display

#plt.figure(figsize=(10,5))
#librosa.display.specshow(librosa.amplitude_to_db(Vft,ref=np.max), x_axis='time', y_axis='log',
#                        sr=sample_rate, hop_length=hop_size)
#plt.colorbar()
#plt.tight_layout()

## Clip STFT
array = librosa.fft_frequencies(sr=sample_rate, n_fft=window_size)
print(array[171])

## Define gausian and triangular filters
import math
sqrt = math.sqrt
pi = math.pi
exp = math.exp

def gaussian(x,mu,sigma=3/8):
    if x<0 : return 0
    return (1/sqrt(2*pi*sigma))*exp(-(((x-mu)**2)/sigma**2))

## Define a single trinagular filter, exponentially distributed
def triangle_filter(f,a,fr=50,r=2):
    # amplitude of the filter is 1
    if(a==0) :
        if(f<0) : return 0
        else : return max(0,1-f/fr)
    elif(a==1) :
        if(f<fr) : return max(0,f/fr)
        else : return max(0,(2*fr-f)/fr)
    else :
        f1 = r**(a-2)*fr
        f2 = r**(a-1)*fr
        f3 = r**a*fr
        if(f<f2) : return max(0,(f-f1)/(f2-f1))
        else : return max(0,(f3-f)/(f3-f2))

## Define K multiples of gaussian filter at the multiples of f0
def multiple_gaussian(K,f,f0,sigma=0.5) :
    if f<0 : return 0
    y=0
    for k in range(K+1): y+=gaussian(f,k*f0,sigma)
    return y

## Define basic block
def basic_block(K,f,a,f0,fr=50,r=2,sigma=0.5) :
    return multiple_gaussian(K,f,f0,sigma)*triangle_filter(f,a,fr,r)

song_index=0
N = len(song_paths)
K = 10
A = 7
S = 4
P = 4
Z = 3
# Hyperparameters
sigma=0.5
fr = 50
r = 2
F = 172

T = 10000
from scipy.io import loadmat
for i in range(len(song_paths)):
    matrix = np.array(loadmat(mats_and_wavs[i][0])['GTF0s']).astype('int')
    T = min(T, matrix[0].shape[0])
print(T)

## Create f0_from_pt
def create_f0_from_pt(f0_from_pt):
    global N,T
    for song_index in range(N):
        matrix = np.array(loadmat(mats_and_wavs[song_index][0])['GTF0s']).astype('int')
        print(matrix.shape)
        for t in range(T):
            index=0
            for p in range(P):
                if matrix[p][t]!=0 and matrix[p][t]<F:
                    f0_from_pt[song_index][t][index]=matrix[p][t]
                    index+=1

f0_from_pt = np.zeros((N,T,P))
create_f0_from_pt(f0_from_pt)

## Update Pt_fbypa
def update_Pt_fbypa(song_index, Pt_fbypa):
    global T,P,A,F,K
    # from p to f0
    for t in range(T):
        print('Updating Pt_fbypa, iteration : '+str(t))
        for p in range(P):
            #print('Updating')
            for a in range(A):
                array = np.array([0.0 for k in range(F)])
                total_basics=0
                for f in range(F):
                    f0=f0_from_pt[song_index,t,p]
                    basic = basic_block(K,f,a,f0,fr,r,sigma)
                    total_basics += basic
                    array[f] = basic
                    #if basic!=0 : print(array[f])
                if total_basics!=0 : array = array/total_basics
                else : array = np.array([0.0 for k in range(F)])
                Pt_fbypa[t,p,a,:]=array
    print('Updated Pt_fbypa')

#Pt_fbypa = np.random.uniform(low=0.1, high=0.9, size=(T,P,A,F))
#update_Pt_fbypa(song_index, Pt_fbypa)

## Save files
directory = '/home/aaditya/store/'
def filepath(filename):
    global directory
    return directory+filename

#np.save(filepath('Pt_fbypa'), Pt_fbypa)
# Pt_p = np.random.uniform(low=0.1, high=0.9, size=(T,P))
# Pt_sbyp = np.random.uniform(low=0.1, high=0.9, size=(T,P,S))
# Pt_zbyps = np.random.uniform(low=0.1, high=0.9, size=(T,P,S,Z))
# P_abysz = np.random.uniform(low=0.1, high=0.9, size=(S,Z,A))

## Normalization rules
def normalize_Pt_p(T,Pt_p) :
    for t in range(T):
        Pt_p[t,:] = Pt_p[t,:]/sum(Pt_p[t,:])

def normalize_Pt_sbyp(T,P,Pt_sbyp) :
    for t in range(T):
        for p in range(P):
            Pt_sbyp[t,p,:] = Pt_sbyp[t,p,:]/sum(Pt_sbyp[t,p,:])

def normalize_Pt_zbyps(T,P,S,Pt_zbyps) :
    for t in range(T):
        for p in range(P):
            for s in range(S):
                Pt_zbyps[t,p,s,:] = Pt_zbyps[t,p,s,:]/sum(Pt_zbyps[t,p,s,:])

def normalize_P_abysz(S,Z,P_abysz) :
    for s in range(S):
        for z in range(Z):
            P_abysz[s,z,:] = P_abysz[s,z,:]/sum(P_abysz[s,z,:])


## Normalize
# normalize_Pt_p(T,Pt_p)
# normalize_Pt_sbyp(T,P,Pt_sbyp)
# normalize_Pt_zbyps(T,P,S,Pt_zbyps)
# normalize_P_abysz(S,Z,P_abysz)


## Update Pt_f

def update_Pt_f(Pt_f,Pt_p,Pt_sbyp,Pt_zbyps,Pt_fbypa,P_abysz) :
    global T,F
    for t in range(T):
        print('Updating Pt_f, iteration : ' + str(t))
        for f in range(F):
            matrix=Pt_p[t,:]*Pt_sbyp[t,:,:]
            matrix=np.reshape(matrix, list(matrix.shape) + [1])
            matrix=matrix*Pt_zbyps[t,:,:,:]
            matrix*=np.tensordot(Pt_fbypa[t,:,:,f],P_abysz[:,:,:],axes=(1,2))
            Pt_f[t,f]=matrix.sum()
    print('Updated Pt_f')

#update_Pt_f(Pt_f,Pt_p,Pt_sbyp,Pt_zbyps,Pt_fbypa,P_abysz)
#print(Pt_f.shape)

## Update rules

def three_dimensional_product(M1,M2):
    A,B,C = M1.shape
    B,C,D = M2.shape
    M3 = np.zeros((A,B,C,D))
    for a in range(A):
        for d in range(D):
            M3[a,:,:,d] = M1[a,:,:]*M2[:,:,d]
    return M3

def four_dimensional_product(M1,M2):
    A,B,C = M1.shape
    B,C,D,E = M2.shape
    M3 = np.zeros((A,B,C,D,E))
    for a in range(A):
        for d in range(D):
            for e in range(E):
                M3[a,:,:,d,e] = M1[a,:,:]*M2[:,:,d,e]
    return M3

def update_Pt_pszabyf(Pt_pszabyf,Pt_f,Pt_p,Pt_sbyp,Pt_zbyps,Pt_fbypa,P_abysz):
    global T,F,P,S,Z,A
    for t in range(T):
        print('Updating Pt_pszabyf, iteration : ' + str(t))
        x = Pt_fbypa[t,:,:,:]/Pt_f[t,:]         #x.shape = (4, 7, 172
        x[np.isnan(x)]=0
        x = np.swapaxes(x,0,2)                  #x.shape = (172, 7, 4)
        x = np.swapaxes(x,1,2)                  #x.shape = (172, 7, 4)
        # x
        y=np.reshape(Pt_p[t,:], list(Pt_p[t,:].shape)+[1])            #y.shape = (4,1)
        y=y*np.ones((1,S))                                            #y.shape = (4,5)
        y=y*Pt_sbyp[t,:,:]                                            #y.shape = (4,5)
        y=np.reshape(y, list(y.shape) + [1])                          #y.shape = (4,5,1)
        y=y*Pt_zbyps[t,:,:,:]                                         #y.shape = (4,5,3)
        y.shape
        #y
        z = three_dimensional_product(y,P_abysz)                      #z.shape = (4,5,3,7)
        z = np.swapaxes(z,1,3)                                        #z.shape = (4,7,3,5)
        #z
        w = four_dimensional_product(x,z)                             #w.shape = (172,4,7,3,5)
        w = np.swapaxes(w,2,4)                                        #w.shape = (172,4,5,3,7)
        Pt_pszabyf[t,:,:,:,:,:]=w
    print('Updated')

## Update rules

def six_dimensional_product(Vft, Pt_pszabyf):
    x = np.reshape(Vft[:F,:T].T, list(Vft[:F,:T].T.shape)+[1,1,1,1])
    x = x*Pt_pszabyf
    return x


def update_Pt_p(Pt_p, Vft_into_Pt_pszabyf):
    # Vft, Pt_pszabyf
    global T, F, P, S, Z, A
    for t in range(T):
        print('Updating Pt_p, iteration : '+str(t))
        total = np.sum(Vft_into_Pt_pszabyf[t, :, :, :, :, :])
        for p in range(P):
            if total != 0:
                Pt_p[t, p] = np.sum(Vft_into_Pt_pszabyf[t, :, p, :, :, :]) / total
            else:
                Pt_p[t, p] = 0
    print('Updated')



def update_Pt_sbyp(Pt_sbyp, Vft_into_Pt_pszabyf):
    # Vft, Pt_pszabyf
    global T, F, P, S, Z, A
    for t in range(T):
        print('Updating Pt_sbyp, iteration : '+str(t))
        for p in range(P):
            # print('Updating')
            total = np.sum(Vft_into_Pt_pszabyf[t, :, p, :, :, :])
            for s in range(S):
                if total != 0:
                    Pt_sbyp[t, p, s] = np.sum(Vft_into_Pt_pszabyf[t, :, p, s, :, :]) / total
                else : Pt_sbyp[t, p, s]=0
    print('Updated')


def update_Pt_zbyps(Pt_zbyps, Vft_into_Pt_pszabyf):
    # Vft, Pt_pszabyf
    global T, F, P, S, Z, A
    for t in range(T):
        print('Updating Pt_zbyps, iteration : '+str(t))
        for p in range(P):
            for s in range(S):
                total = np.sum(Vft_into_Pt_pszabyf[t, :, p, s, :, :])
                for z in range(Z):
                    if total != 0:
                        Pt_zbyps[t, p, s, z] = np.sum(Vft_into_Pt_pszabyf[t, :, p, s, z, :]) / total
                    else : Pt_zbyps[t, p, s, z]=0
    print('Updated')


def update_P_abysz(P_abysz, Vft_into_Pt_pszabyf):
    global T, F, P, S, Z, A
    for s in range(S):
        for z in range(Z):
            print('Updating P_abysz, iteration : ' + str(s)+str(z))
            total = np.sum(Vft_into_Pt_pszabyf[:, :, :, s, z, :])
            for a in range(A):
                if total != 0:
                    P_abysz[s, z, a] = np.sum(Vft_into_Pt_pszabyf[:, :, :, s, z, :]) / total
                else:
                    P_abysz[s, z, a] = 0
    print('Updated')


# Create Pt_fbys
def create_Pt_fbys(Pt_fbys, Pt_fbypa, Pt_p, Pt_sbyp, Pt_zbyps, P_abysz):
    global T,S,F,P,A
    for t in range(T):
        print('Creating Pt_fbys, iteration : ' + str(t))
        for s in range(S):
            for f in range(F):
                matrix=Pt_p[t,:]*Pt_sbyp[t,:,s]
                matrix=np.reshape(matrix, list(matrix.shape) + [1])
                matrix=matrix*Pt_zbyps[t,:,s,:]
                matrix*=np.tensordot(Pt_fbypa[t,:,:,f],P_abysz[s,:,:],axes=(1,1))
                Pt_fbys[t,s,f]=matrix.sum()
    print("Created Pt_fbys")

# Create Vft_s
def create_Vft_s(Vft_s, Vft, Pt_fbys, Pt_f):
    global S
    for s in range(S):
        Vft_s[s,:,:]=(Pt_fbys[:,s,:]/Pt_f).T*Vft[:F,:T]
    Vft_s[np.isnan(Vft_s)]=0


## Update
def update(upto_index, max_iter):
    global mats_and_wavs,T,F,P,S,Z,A
    for i in range(upto_index):
        ## Spectrogram

        song, sample_rate = librosa.load(mats_and_wavs[i][1])
        Stft = librosa.stft(song, n_fft=window_size, hop_length=hop_size)
        Vft, phase = librosa.magphase(Stft)
        #Pt_fbypa = np.random.uniform(low=0.1, high=0.9, size=(T,P,A,F))
        #update_Pt_fbypa(song_index, Pt_fbypa)
        Pt_fbypa = np.load(filepath('Pt_fbypa_')+str(i)+'.npy')
        ## Initialize

        Pt_f = np.random.uniform(low=0.1, high=0.9, size=(T, F))
        Pt_p = np.random.uniform(low=0.1, high=0.9, size=(T, P))
        Pt_sbyp = np.random.uniform(low=0.1, high=0.9, size=(T, P, S))
        Pt_zbyps = np.random.uniform(low=0.1, high=0.9, size=(T, P, S, Z))
        P_abysz = np.random.uniform(low=0.1, high=0.9, size=(S, Z, A))
        Pt_fbys = np.random.uniform(low=0.1, high=0.9, size=(T, S, F))
        Vft_s = np.random.uniform(low=0.1, high=0.9, size=(S, F, T))

        ## Normalize

        normalize_Pt_p(T, Pt_p)
        normalize_Pt_sbyp(T, P, Pt_sbyp)
        normalize_Pt_zbyps(T, P, S, Pt_zbyps)
        normalize_P_abysz(S, Z, P_abysz)

        ## Update

        Pt_pszabyf = np.random.uniform(low=0.1, high=0.9, size=(T, F, P, S, Z, A))
        for iteration in range(max_iter) :
            # E_step
            print(iteration)
            update_Pt_pszabyf(Pt_pszabyf,Pt_f,Pt_p,Pt_sbyp,Pt_zbyps,Pt_fbypa,P_abysz)
            Vft_into_Pt_pszabyf = six_dimensional_product(Vft, Pt_pszabyf)
            # M_step
            update_Pt_p(Pt_p, Vft_into_Pt_pszabyf)
            update_Pt_sbyp(Pt_sbyp, Vft_into_Pt_pszabyf)
            update_Pt_zbyps(Pt_zbyps, Vft_into_Pt_pszabyf)
            update_P_abysz(P_abysz, Vft_into_Pt_pszabyf)
            update_Pt_f(Pt_f,Pt_p,Pt_sbyp,Pt_zbyps,Pt_fbypa,P_abysz)

        # Save

        #np.save(filepath('Pt_fbypa_') + str(i) + '.npy', Pt_fbypa)
        np.save(filepath('Pt_pszabyf_')+str(i)+'.npy', Pt_pszabyf)
        np.save(filepath('Pt_f_')+str(i)+'.npy', Pt_f)
        np.save(filepath('Pt_p_')+str(i)+'.npy', Pt_p)
        np.save(filepath('Pt_sbyp_')+str(i)+'.npy', Pt_sbyp)
        np.save(filepath('Pt_zbyps_')+str(i)+'.npy', Pt_zbyps)
        np.save(filepath('P_abysz_')+str(i)+'.npy', P_abysz)

        # Create Pt_fbys and Vft_s

        create_Pt_fbys(Pt_fbys, Pt_fbypa, Pt_p, Pt_sbyp, Pt_zbyps, P_abysz)
        np.save(filepath('Pt_fbys_')+str(i)+'.npy', Pt_fbys)
        create_Vft_s(Vft_s, Vft, Pt_fbys, Pt_f)
        np.save(filepath('Vft_s_')+str(i)+'.npy', Vft_s)

# Call
update(len(mats_and_wavs),2)



