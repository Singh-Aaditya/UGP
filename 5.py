##### MIREX-05 DATA GENERATOR AND CLASSIFIER #####

import numpy as np
import pandas as pd

'''''
#### Train Input ####
train_examples=28
def file_path(i) :
    return '/home/aaditya/MIREX_05/Train/train' + str(i) + '.wav'

data = [0 for k in range(train_examples)]
from scipy.io import wavfile
for i in range(1,train_examples+1) :
    sample_rate, data[i-1] = wavfile.read(file_path(i))

## Compute STFT
Stft = [0 for k in range(train_examples)]
sample_frequency = [0 for k in range(train_examples)]
segment_time = [0 for k in range(train_examples)]

from scipy.signal import stft
for i in range(train_examples) :
    sample_frequency[i], segment_time[i], Stft[i] = np.array(stft(data[i], fs=0.064, window='hamming', noverlap=0.75))

#minimum=np.float('inf');   minimum = 2398
max_log_stft = -np.float('inf')
log_stft = np.zeros((train_examples,129,2397))
for i in range(train_examples) :
    print('Calculating')
    for j in range(129) :
        # minimum = min(minimum, len(Stft[i][j]))
        for k in range(2397) :
             log_stft[i,j,k] = np.abs(np.log(np.abs(Stft[i][j][k])))
             max_log_stft = max(max_log_stft, log_stft[i,j,k])

# Normalize
log_stft /= max_log_stft            # rescale between 0 and 1
smaller_log_stft = log_stft.reshape(train_examples, 309213)
dataframe = pd.DataFrame(smaller_log_stft)
dataframe.to_csv('/home/aaditya/MIREX_05/train_log_stft.csv')
'''''


#### MIREX-05 CLASSIFIER TRAINING ####
train_examples=28
dataframe = pd.read_csv('./MIREX_05/train_log_stft.csv')
log_stft = dataframe.iloc[:].values
log_stft = np.delete(log_stft, 0, 1)    # remove the column of indices
log_stft = log_stft.reshape(train_examples,129,2397)
log_stft = log_stft[:,:,:2391]

## Output
def file_path(i) :
    return './MIREX_05/Train/train' + str(i) + '.txt'

dataframe_1 = pd.read_table(file_path(1), delim_whitespace=True, names=("0th", "1st"))
def column_name(i) :
    if(i==2) : return "2nd"
    elif(i==3): return "3rd"
    else : return str(i) + "th"

for i in range(2,train_examples) :
    dataframe_2 = pd.read_table(file_path(i), delim_whitespace=True, names=("0th", column_name(i)))
    dataframe_1 = pd.merge(dataframe_1, dataframe_2, how='outer')

print(dataframe_1.shape)
dataframe_1.fillna(0, inplace=True)
#dataframe_1.dropna(axis=0, inplace=True)     # remove NaN
print(dataframe_1.shape)

import numpy as np
output_data = dataframe_1[:2391].values.astype('int')
output_data = np.delete(output_data, 0, 1)

# Classification
def round_off(x, precision=5, base=2.5) : return (base*((x/base).round())).round(precision)
output_data = round_off(output_data)    # smallest=48; largest=1052; number_of_classes=192 + 1;
def into_classes(output_data):
    for i in range(output_data.shape[0]) :
        for j in range(output_data.shape[1]) :
            if (output_data[i,j]==0.0) : output_data[i,j]=0
            else : output_data[i,j] = 1 + int((output_data[i,j]-48)/5.26)
into_classes(output_data)

## Training


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.regularizers import l2
classifier = Sequential()
classifier.add(Dense(500, input_dim=129, activation='relu', kernel_regularizer = l2(0.01)))
#classifier.add(Dropout(0.5))
classifier.add(Dense(500, activation='relu', kernel_regularizer = l2(0.01)))
#classifier.add(Dropout(0.5))
classifier.add(Dense(193, activation = 'softmax'))

from tensorflow.keras.optimizers import Adam
adam = Adam(lr = pow(10,-3), beta_1=0.9, beta_2=0.999)
classifier.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['mae','acc'])

x_train = log_stft[:27,:,:2391]
x_train = np.reshape(x_train , (27*2391,129))
y_train = output_data.T
y_train = np.reshape(y_train , (27*2391,))
y_train = to_categorical(y_train, 193)
classifier.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.4,   verbose = 1)
classifier.save('./store/classifier.h5')


'''''
#### Test Input ####
test_examples = 3
def file_path(i) : return './MIREX_05/Test/test' + str(i) + '.wav'

data = [0 for k in range(test_examples)]
from scipy.io import wavfile
for i in range(1,test_examples+1) :
    sample_rate, data[i-1] = wavfile.read(file_path(i))

## Compute STFT
Stft = [0 for k in range(test_examples)]
sample_frequency = [0 for k in range(test_examples)]
segment_time = [0 for k in range(test_examples)]

from scipy.signal import stft
for i in range(test_examples) :
    sample_frequency[i], segment_time[i], Stft[i] = np.array(stft(data[i], fs=0.064, window='hamming', noverlap=0.75))

max_log_stft = -np.float('inf')
log_stft = np.zeros((test_examples,129,2397))
for i in range(test_examples) :
    print('Calculating')
    for j in range(129) :
        for k in range(2397) :
            log_stft[i,j,k] = np.abs(np.log(np.abs(Stft[i][j][k])))
            max_log_stft = max(max_log_stft, log_stft[i,j,k])


# Normalize
log_stft /= max_log_stft            # rescale between 0 and 1
smaller_log_stft = log_stft.reshape(3, 309213)
dataframe = pd.DataFrame(smaller_log_stft)
dataframe.to_csv('./MIREX_05/test_log_stft.csv')
'''''

'''''
#### MIREX-05 CLASSIFIER TESTING ####

test_examples = 3
# Predefined functions
def column_name(i) :
    if(i==2) : return "2nd"
    elif(i==3): return "3rd"
    else : return str(i) + "th"

def round_off(x, precision=5, base=2.5) : return (base*((x/base).round())).round(precision)

## Load model and test input
from tensorflow.keras.models import load_model
classifier = load_model('/home/aaditya/store/classifier.h5')
dataframe = pd.read_csv('./MIREX_05/test_log_stft.csv')
log_stft_2 = dataframe.iloc[:].values
log_stft_2 = np.delete(log_stft_2, 0, 1)    # remove the column of indices
log_stft_2 = log_stft_2.reshape(3,129,2397)

## Output
def file_path_2(i) : return './MIREX_05/Test/test' + str(i) + '.txt'


dataframe_1 = pd.read_table(file_path_2(1), delim_whitespace=True, names=("0th", "1st"))
for i in range(2,test_examples+1) :
    dataframe_2 = pd.read_table(file_path_2(i), delim_whitespace=True, names=("0th", column_name(i)))
    dataframe_1 = pd.merge(dataframe_1, dataframe_2, how='outer')
dataframe_1.dropna(axis=0, inplace=True)     # remove NaN

output_data_2 = dataframe_1.iloc[:2391].values
output_data_2 = np.delete(output_data_2, 0, 1)

output_data_2 = round_off(output_data_2)    # smallest =48; largest=1092; number_of_classes=192 + 1;
def into_classes(output_data):
    for i in range(output_data.shape[0]) :
        for j in range(output_data.shape[1]) :
            if (output_data[i,j]==0.0) : output_data[i,j]=0
            else : output_data[i,j] = 1 + int((output_data[i,j]-48)/5.26)
into_classes(output_data_2)

from tensorflow.keras.utils import to_categorical
x_test = log_stft_2[:3,:,:2391]
x_test = np.reshape(x_test , (3*2391,129))
y_test = output_data_2.T
y_test = np.reshape(y_test , (3*2391,))
y_test = to_categorical(y_test, 193)
classifier.evaluate(x_test, y_test, batch_size=4, verbose = 1)
'''''
