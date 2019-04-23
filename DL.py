# In[1]:
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

## Create song paths
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
import IPython
from scipy.io import loadmat

dataset_path = '/home/aaditya/Bach10/'
def create_song_paths(dataset_path):
    song_paths=[]
    for song_name in os.listdir(dataset_path) :
        if (song_name != '.DS_Store') : song_paths.append(dataset_path + song_name + '/')
    return sorted(song_paths)

song_paths = create_song_paths(dataset_path)
#print(song_paths)

## Create mats_and_wavs
list_of_instruments = ['Violin', 'Clarinet', 'Saxophone','Bassoon']
def create_mats_and_wavs(dataset_path, song_paths):
    mats_and_wavs = [[] for k in range(len(song_paths))]
    length = len(dataset_path)
    for i in range(len(song_paths)):
        mats_and_wavs[i].append(song_paths[i]+song_paths[i][length:-1]+'-violin.wav')
        mats_and_wavs[i].append(song_paths[i]+song_paths[i][length:-1]+'-clarinet.wav')
        mats_and_wavs[i].append(song_paths[i]+song_paths[i][length:-1]+'-saxphone.wav')
        mats_and_wavs[i].append(song_paths[i]+song_paths[i][length:-1]+'-bassoon.wav')
        mats_and_wavs[i].append(song_paths[i]+song_paths[i][length:-1]+'.wav')
        mats_and_wavs[i].append(song_paths[i]+song_paths[i][length:-1]+'-GTF0s.mat') 
    return mats_and_wavs

mats_and_wavs = create_mats_and_wavs(dataset_path, song_paths)
print(mats_and_wavs[0][4])

## Read wavfile for a single song 
song, sample_rate = librosa.load(mats_and_wavs[0][4])
#IPython.display.Audio(data = song, rate = sample_rate)
# sample_rate = 22050


# In[2]:


## Plot spectrogram for a single complete song
window_size = int(0.084*sample_rate)
hop_size = int(0.01*sample_rate)
# Stft = librosa.stft(song, n_fft = window_size, hop_length=hop_size)
# Vft, phase = librosa.magphase(Stft)
# #Plot
# plt.figure(figsize=(10,5))
# librosa.display.specshow(librosa.amplitude_to_db(Vft,ref=np.max), x_axis='time', y_axis='log',
#                         sr=sample_rate, hop_length=hop_size)
# plt.colorbar()
# plt.tight_layout()


# In[3]:


## Define a function to lose information from a matrix
def create_reduced_matrix(matrix):
    S = len(matrix)
    T = len(matrix[0])
    reduced_matrix = [[] for k in range(T)]
    #print(reduced_matrix)
    for i in range(T):
        for j in range(S):
            if (matrix[j][i]>0) : reduced_matrix[i].append(matrix[j][i])
        reduced_matrix[i] = sorted(reduced_matrix[i])
    return reduced_matrix

x=[[10,0],[30,0],[15,25],[0,40]]
z = create_reduced_matrix(x)
print(z)
# z[i] empty when no instrument is playing


# In[4]:


## Load Pt_fbypa
directory = '/home/aaditya/store/'
def filepath(filename):
    global directory
    return directory+filename

Pt_fbypa=np.load(filepath('Pt_fbypa_0')+'.npy')
print(Pt_fbypa.shape)
print(np.max(Pt_fbypa), np.min(Pt_fbypa)) 
T,S,A,F = Pt_fbypa.shape


# In[14]:


## Create the Data Loader

# Create a list of reduced matrices
def create_list_of_reduced_matrices(mats_and_wavs):
    global T
    list_of_reduced_matrices = []
    for i in range(len(mats_and_wavs)):
        matrix = np.array(loadmat(mats_and_wavs[i][5])['GTF0s']).astype('int')
        T = min(T, matrix.shape[1])
        reduced_matrix = create_reduced_matrix(matrix)
        list_of_reduced_matrices.append(reduced_matrix)
    return np.array(list_of_reduced_matrices)


def create_list_of_Pt_fbys(song_paths):
    list_of_Pt_fbys = []
    for i in range(len(song_paths)):
        Pt_f=np.load(filepath('Pt_f_')+str(i)+'.npy')    # shape is (T,F)
        Pt_pszabyf=np.load(filepath('Pt_pszabyf_')+str(i)+'.npy')    # shape is (T,F,P,S,Z,A)
        matrix=np.reshape(Pt_f, list(Pt_f.shape)+[1,1,1,1])*Pt_pszabyf
        print(i, Pt_f.shape, Pt_pszabyf.shape)
        # sum over p,z,a
        matrix=np.sum(matrix, (2,4,5))    # shape is (T,F,S)
        matrix=np.swapaxes(matrix,0,2)    # shape is (S,F,T)
        list_of_Pt_fbys.append(matrix)
    return np.array(list_of_Pt_fbys)


#list_of_red_mat = create_list_of_reduced_matrices(mats_and_wavs)
#list_of_Pt_fbys = create_list_of_Pt_fbys(song_paths)
#np.save(filepath('list_of_red_mat')+'.npy', list_of_red_mat)
#np.save(filepath('list_of_Pt_fbys')+'.npy', list_of_Pt_fbys)
list_of_red_mat = np.load(filepath('list_of_red_mat')+'.npy')
list_of_Pt_fbys = np.load(filepath('list_of_Pt_fbys')+'.npy')


# Create a function to return a one hot vector corresponding to the source at a given p and t
from tensorflow.keras.utils import to_categorical
def one_hot_output(mats_and_wavs, song_path_index, t, p):
    global S
    matrix = np.array(loadmat(mats_and_wavs[song_path_index][5])['GTF0s']).astype('int')
    count=0
    array = sorted(matrix[:,t])
    for s in range(S):
        if array[s]==0 : continue
        else : 
            if p==count : break
            else : count+=1           
    if s<S : 
        label = np.array([s])
        label = to_categorical(label, num_classes = S)
        return label


# In[ ]:


# Create generator
def generator(list_of_songs, batch_size):
    global Pt_fbypa, mats_and_wavs, list_of_red_mat, list_of_Pt_fbys, T, A, F
    iterate=0
    while True:
        #print(iterate)
        for song_path_index in list_of_songs :
            red_mat = list_of_red_mat[song_path_index]
            Pt_fbys = list_of_Pt_fbys[song_path_index]
            Pt_fbypa = np.load(filepath('Pt_fbypa_'+str(song_path_index))+'.npy')
            # shape of Pt_fbys is [S,F,T]
            batch=0
            for t in range(T):
                for p in range(len(red_mat[t])):
                    if batch<batch_size:
                        input1 = Pt_fbypa[t,p,:,:]
                        input1 = np.reshape(input1, [1]+list(input1.shape)+[1])
                        input2 = Pt_fbys[:,:,t]
                        input2 = np.reshape(input2, [1] + list(input2.shape))
                        outputs = one_hot_output(mats_and_wavs, song_path_index, t, p)
                        if batch==0 : 
                            batch_input1 = input1
                            batch_input2 = input2
                            batch_outputs = outputs
                        else : 
                            batch_input1 = np.concatenate((batch_input1, input1), axis=0)
                            batch_input2 = np.concatenate((batch_input2, input2), axis=0)
                            batch_outputs = np.concatenate((batch_outputs, outputs), axis=0)
                        #print(batch_input1.shape, batch_input2.shape, batch_outputs.shape)
                        batch+=1
                    else : 
                        batch=0
                        inputs = [batch_input1, batch_input2]
                        outputs = batch_outputs
                        yield (inputs, outputs)
        #iterate+=1
list_of_train_songs = [0,1,3,4,5,6,7,9]
list_of_val_songs = [2,8]

def total_examples(list_of_songs):
    total=0
    for song_index in list_of_songs:
        for i in range(len(list_of_red_mat[song_index])):
            total+=len(list_of_red_mat[song_index][i])
    return total

train_samples = total_examples(list_of_train_songs)
val_samples = total_examples(list_of_val_songs)
print(train_samples)
print(val_samples)
train_generator = generator(list_of_train_songs, 32)
val_generator = generator(list_of_val_songs, 16)


# In[ ]:


## Define a function to avoid NaN gradients
import tensorflow as tf


def inverse_without_nans(x):
    f = lambda x: 1.0 / x
    non_zero_mask = tf.not_equal(x, 0.0)
    x_zeros_replaced_by_ones = tf.where(non_zero_mask, x, tf.ones_like(x))
    y_replaced_by_zeros = tf.zeros_like(x)
    y_nans_replaced_by_zeros = tf.where(non_zero_mask, f(x_zeros_replaced_by_ones),
                                        y_replaced_by_zeros)
    return y_nans_replaced_by_zeros


## Define custom model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D
from tensorflow.keras.regularizers import l2


class custom_model(tf.keras.Model):
    global A, F

    def __init__(self):
        super(custom_model, self).__init__()
        self.conv1 = Conv2D(input_shape=(A, F, 1), filters=2, kernel_size=(3, 3),
                            strides=(1, 2), activation='relu',
                            kernel_regularizer=l2(0.001))
        self.conv2 = Conv2D(filters=4, kernel_size=(2, 2), strides=(1, 1), activation='relu',
                            kernel_regularizer=l2(0.001))
        self.flatten = Flatten()
        self.dense1 = Dense(F, activation='softmax', kernel_regularizer=l2(0.001))

    def call(self, inputs):
        Pt_fbypa_given_tp = inputs[0]
        Pt_fbys = inputs[1]
        # print(Pt_fbypa_given_tp.shape, Pt_fbys_given_t.shape)
        x = self.conv1(Pt_fbypa_given_tp)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        # print(x.shape)
        x = tf.expand_dims(x, 1)
        x = Pt_fbys * inverse_without_nans(x)
        # x = tf.where(tf.math.is_nan(x), tf.ones_like(x)*0.0001, x)
        # print(x.shape)
        x = tf.reduce_mean(x, axis=-1)
        # print(x.shape)
        # x=tf.clip_by_value(x, 0.001, 0.999)
        return x


model = custom_model()
# Compile
from tensorflow.keras.optimizers import Adam

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['mae', 'acc'])

# Train
model.fit_generator(generator=train_generator, epochs=10, steps_per_epoch=train_samples / 32,
                    validation_data=val_generator, validation_steps=val_samples / 16,
                    verbose=1, shuffle=True)

# Save weights
model.save_weights('/home/aaditya/store/DL_weights.h5')



