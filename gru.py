from augmentation import resample_augment
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from lstm_utils import temporalize, flatten, scale
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TruePositives, TrueNegatives, Recall 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, GRU, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
"""### Load Data"""

data_path = 'data/processminer-rare-event-detection-data-augmentation.xlsx'
data_file = pd.ExcelFile(data_path)
data = pd.read_excel(data_file, 'data-(b)-4-min-ahead-conse-rmvd')

"""### Train-Test-Validation Split"""

SEED = 0
DATA_SPLIT_PCT = 0.10
label_name = 'y-4min-ahead'
LOOKBACK=20

data = data.drop(['time', 'x28', 'x61'], axis=1)

input_X = data.loc[:, data.columns != label_name].values  # converts the df to a numpy array
input_y = data[label_name].values

n_features = input_X.shape[1]  # number of features

X, y = temporalize(input_X, input_y, LOOKBACK)


X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=DATA_SPLIT_PCT, random_state=SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=DATA_SPLIT_PCT, random_state=SEED)

print(X_train.shape, y_train.shape)

X_train = X_train.reshape(X_train.shape[0], LOOKBACK, n_features)
X_valid = X_valid.reshape(X_valid.shape[0], LOOKBACK, n_features)
X_test = X_test.reshape(X_test.shape[0], LOOKBACK, n_features)

print(X_train.shape, y_train.shape)

X_train, y_train = resample_augment(X_train, y_train)
X_train, y_train = shuffle(X_train, y_train, random_state=SEED)
print(X_train.shape, y_train.shape)

scaler = StandardScaler().fit(flatten(X_train))
X_train_scaled = scale(X_train, scaler)
X_valid_scaled = scale(X_valid, scaler)
X_test_scaled = scale(X_test, scaler)

print(X_train.shape, y_train.shape)

epochs = 200
batch = 128
lr = 0.0001

lstm_model = Sequential()
lstm_model.add(GRU(units=128, activation='relu', return_sequences=False,
               dropout = 0.5, recurrent_dropout=0.5, input_shape=(LOOKBACK, n_features)))
#lstm_model.add(LSTM(units=256, activation='relu', return_sequences=False))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=lr),
                   metrics=['accuracy', FalsePositives(), FalseNegatives(), TruePositives(), TrueNegatives(), Recall()])
print(lstm_model.summary())

from tensorflow.keras.utils import to_categorical

lstm_model.fit(X_train_scaled, y_train, epochs=epochs,
               batch_size=batch,
               validation_data=(X_valid_scaled, y_valid),
               verbose=2)

