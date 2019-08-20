from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from lstm_utils import temporalize, flatten, scale
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
"""### Load Data"""

data_path = 'data/processminer-rare-event-detection-data-augmentation.xlsx'
data_file = pd.ExcelFile(data_path)
data = pd.read_excel(data_file, 'data-(b)-4-min-ahead-conse-rmvd')

"""### Train-Test-Validation Split"""

SEED = 0
DATA_SPLIT_PCT = 0.10
label_name = 'y-4min-ahead'
LOOKBACK=15

data = data.drop(['time', 'x28', 'x61'], axis=1)

input_X = data.loc[:, data.columns != label_name].values  # converts the df to a numpy array
input_y = data[labe_name].values

n_features = input_X.shape[1]  # number of features

X, y = temporalize(input_X, input_y, LOOKBACK)


X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=DATA_SPLIT_PCT, random_state=SEED)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=DATA_SPLIT_PCT, random_state=SEED)

X_train = X_train.reshape(X_train.shape[0], LOOKBACK, n_features)
X_valid = X_valid.reshape(X_valid.shape[0], LOOKBACK, n_features)
X_test = X_test.reshape(X_test.shape[0], LOOKBACK, n_features)

scaler = StandardScaler().fit(flatten(X_train))
X_train_scaled = scale(X_train, scaler)
X_valid_scaled = scale(X_valid, scaler)
X_test_scaled = scale(X_test, scaler)

epochs = 200
batch = 64
lr = 0.0001




