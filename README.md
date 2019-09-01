# Rare Event Classification Challenge

### Overview of the Approach

My solution to the challenge combines data augmentation, feature engineering and feature learning.

  

### Preprocessing

First off, the categorical and binary features are removed from the data. 

  

Since the problem is one of predicting if an event will happen in 4 minutes and not whether the event has already happened, I changed the labels of 2 and 4 minutes before any event (positive sample) to positive, and then removed the positive sample itself.

  

Data is transformed into 3-dimensional to later be fed into recurrent neural network models. A window size of 20 seems to have the best performance among window sizes that I tried.

  

The data is then divided into training (80%) and testing sets (20%).

  

Each window of training and testing data is then normalized using z-score normalization based on statistics learned from training set only.

### Data Augmentation 

Due to severe class imbalance with the positive class making up less than 1% of the data, either resampling the positive class or assigning large weights to the positive class loss are necessary, otherwise any machine learning model would learn to always predict negative (no event).

  

To assign a proper weight to the positive class loss, I measured to proportion of negative to positive data available in training set, and learned that there are 69 times more negative samples than positive. I, hence, used weight 69 for the positive class loss. Alternatively, I repeated the positive samples in the dataset 69 times.

  

Both of these approaches increase the overall gradient size resulting from positive samples; the first approach does this via single big gradient steps while the second approach results in many smaller gradient descent steps. From a non-convex optimization perspective, several smaller steps are more likely to converge to a local minimum than giant steps, and this is confirmed by superior results we achieved from resampling compared to providing class weights.

### Feature Engineering

I augmented the data by adding several new features to the set of features. I used the first order gradients of the sensor outputs with respect to time as features, doubling the number of features. These features are meant to provide the predictive model with insights into the speed of change in sensor values, a concept that might take a lot of data and training for a model to learn on itself.

After achieving positive results, I added second order gradients to the feature set as well, observing even more improvement.

### Feature Learning

I used recurrent neural networks with GRU cells to learn time-aware features. I use sequences of length 20 to train GRU models, and use the representation learned by the model to transform the test data.



### Submitted Code

This repository contains a test.py script that expects the following flags:
 
 - data_path: path to test data.
 
 - sheet name: if the data file is a multi-sheet excel file, provide sheet name here.
 
 - label name: Name of label column in data file.
 
 - data_scaler_path: path to pickled Scaler object created based on training data.
 
 - final_model_path: path to final Keras model learned from training data to transform the test data.

The script does the following:

1. Reads the held-out data assuming it is in the exact same format as the training data (excel sheets),

2. Normalizes the held-out data using statistics learned from the training data,

3. Preprocesses the held-out data to prepare for input to the trained model,

4. Loads the final trained model, and feed the held-out data to it.

5. The output of the last recurrent layer of trained model is then fed to the logistic regression model required by the challenge. Namely, sklearn.linear_model.LogisticRegression with parameters class_weight = ’balanced’, max_iter = 10000, penalty = ’l1’, solver = ’saga’, C = 0.01.

6. The predictions from logistic regression are evaluated using a number of metrics: accuracy, precision, recall, and F-score.
