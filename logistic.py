from augmentation import *
from data_utils import prepare_data
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

AUGMENT = 'NO'

"""## Helper Methods"""

def scale_datasets(scaler, train_set, to_scale):
  scaler = scaler().fit(train_set)
  res = []
  for dataset in to_scale:
    res.append(scaler.transform(dataset))
  return res

def display_results(precision_recall_fscore_support):
  disp_df = pd.DataFrame(data=precision_recall_fscore_support, columns=['Pos Class'], index=['Precision', 'Recall', 'F-score', 'support'])
  print(disp_df)


X, y = prepare_data()

"""### Train-Test-Validation Split"""
SEED = 0
DATA_SPLIT_PCT = 0.20

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=DATA_SPLIT_PCT, random_state=SEED)

classifier = LogisticRegression(class_weight='balanced', max_iter = 10000, penalty = 'l1', solver = 'saga', C = 0.01, verbose=1)

"""## No Augmentation"""
train_x_scaled, test_x_scaled = scale_datasets(StandardScaler, X_train, [X_train, X_test])

if AUGMENT == 'NO':
    classifier.fit(train_x_scaled, y_train)
    y_hat_train = classifier.predict(train_x_scaled)
    train_res = precision_recall_fscore_support(y_train, y_hat_train)#, average='binary')

elif AUGMENT == 'GRAD':
    augmented_train_x, augmented_y_train = gradient_augment(train_x_scaled, y_train)
    augmented_train_x_rescaled, augmented_y_train = shuffle(augmented_train_x, augmented_y_train)

    classifier.fit(augmented_train_x_rescaled, augmented_y_train)
    y_hat_train = classifier.predict(augmented_train_x_rescaled)
    train_res = precision_recall_fscore_support(augmented_y_train, y_hat_train, average='binary')
    test_x_scaled, _ = gradient_augment(test_x_scaled, y_test)

elif AUGMENT == 'GRADRESAMP':
    augmented_train_x, augmented_y_train = gradient_augment(train_x_scaled, y_train)
    augmented_train_x, augmented_y_train = resample_augment(augmented_train_x, augmented_y_train)
    augmented_train_x, augmented_y_train = shuffle(augmented_train_x, augmented_y_train)
    augmented_train_x_rescaled = scale_datasets(StandardScaler, augmented_train_x, [augmented_train_x])[0]
    classifier.fit(augmented_train_x_rescaled, augmented_y_train)
    y_hat_train = classifier.predict(augmented_train_x_rescaled)
    train_res = precision_recall_fscore_support(augmented_y_train, y_hat_train, average='binary')
    test_x_scaled, _ = gradient_augment(test_x_scaled, y_test)

else:
    augmented_train_x, augmented_y_train = resample_augment(df_train_x, y_train)

    augmented_train_x_rescaled = scale_datasets(StandardScaler, augmented_train_x, [augmented_train_x])[0]

    augmented_train_x_rescaled, test_x_scaled = scale_datasets(StandardScaler, augmented_train_x, [augmented_train_x, df_test_x])
    classifier.fit(augmented_train_x_rescaled, augmented_y_train)
    y_hat_train = classifier.predict(augmented_train_x_rescaled)
    train_res = precision_recall_fscore_support(augmented_y_train, y_hat_train, average='binary')

#display_results(train_res)
print(train_res)

y_hat_test = classifier.predict(test_x_scaled)
test_res = precision_recall_fscore_support(y_test, y_hat_test)#, average='binary')
#display_results(test_res)
print(test_res)
test_res = confusion_matrix(y_test, y_hat_test, labels=[1, 0])
print(test_res)
