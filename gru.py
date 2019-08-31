from augmentation import resample_augment, gradient_augment 
from data_utils import label_shift, prepare_data
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from time_series_utils import temporalize, flatten, scale
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TruePositives, TrueNegatives, Recall, Precision 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, GRU, LSTM, Reshape

SEED = 0
DATA_SPLIT_PCT = 0.20
LOOKBACK=20
ALL_SEQ=True
AUGMENT_METHOD = 'resample_and_augment'

input_X, input_y = prepare_data()
if 'gradient' in AUGMENT_METHOD:
    input_X, input_y = gradient_augment(input_X, input_y, order=1)

n_features = input_X.shape[1]

X, y = temporalize(input_X, input_y, LOOKBACK)

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=DATA_SPLIT_PCT, random_state=SEED)
#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=DATA_SPLIT_PCT, random_state=SEED)

X_train = X_train.reshape(X_train.shape[0], LOOKBACK, n_features)
#X_valid = X_valid.reshape(X_valid.shape[0], LOOKBACK, n_features)
X_test = X_test.reshape(X_test.shape[0], LOOKBACK, n_features)


if 'resample' in AUGMENT_METHOD:
    X_train, y_train = resample_augment(X_train, y_train)

X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

print(X_train.shape, X_test.shape)

scaler = StandardScaler().fit(flatten(X_train))
X_train_scaled = scale(X_train, scaler)
#X_valid_scaled = scale(X_valid, scaler)
X_test_scaled = scale(X_test, scaler)


epochs = 400
batch = 128
lr = 0.0001
hidden_size = 128

gru_model = Sequential()
if ALL_SEQ == True:
    gru_model.add(GRU(units=hidden_size, activation='relu', return_sequences=True,
                   dropout = 0.5, recurrent_dropout=0.5, input_shape=(LOOKBACK, n_features)))
    gru_model.add(Reshape((LOOKBACK*hidden_size,), input_shape=(LOOKBACK, hidden_size)))
else:
    gru_model.add(GRU(units=hidden_size, activation='relu', return_sequences=False,
                   dropout = 0.5, recurrent_dropout=0.5, input_shape=(LOOKBACK, n_features)))
gru_model.add(Dense(1, activation='sigmoid'))
gru_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=lr, decay=0.0),
                   metrics=['accuracy', FalsePositives(), FalseNegatives(), 
                            TruePositives(), TrueNegatives(), 
                            Recall(), Precision()])
print(gru_model.summary())

if 'resample' not in AUGMENT_METHOD:
    class_weight = {0: 1.,
                    1: 69.}
else:
    class_weight = {0: 1.,
                    1: 1.}
gru_model.fit(X_train_scaled, y_train, epochs=epochs,
               batch_size=batch,
               validation_data=(X_test_scaled, y_test),
               validation_freq=100,
               verbose=2, class_weight=class_weight)

scores = gru_model.evaluate(X_test_scaled, y_test, verbose=1)
print(scores)

# Save the model
gru_model.save('final_model.ker')
X = np.array(X)
X = X.reshape(X.shape[0], LOOKBACK, n_features)
final_scaler = StandardScaler().fit(flatten(X))
with open('final_scaler.pickle', 'wb') as wf:
    pickle.dump(final_scaler, wf)
