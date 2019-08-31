from absl import flags 
from absl import app
from augmentation import gradient_augment
from data_utils import prepare_data
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from time_series_utils import temporalize, flatten, scale
import tensorflow as tf
import tensorflow.keras as keras

flags.DEFINE_string('data_path', 'data/processminer-rare-event-detection-data-augmentation.xlsx', 'Path to test data.')
flags.DEFINE_string('sheet_name', 'data-(a)-raw-data', 'Sheet name, in case the input file is a multi-sheet excel file.')

flags.DEFINE_string('label_name', 'y', 'Name of label column.')

flags.DEFINE_string('output_path', './logs/x', 'Path to save logs and other outputs.')

flags.DEFINE_string('final_model_path', 'final_model.ker', 'Path to trained model.')

flags.DEFINE_string('data_scaler_path', 'final_scaler.pickle', 'Path to scaler for data normalization.')

FLAGS = flags.FLAGS
LOOKBACK = 20

def main(argv):
    X, y = prepare_data(data_path=FLAGS.data_path,
                                  sheet_name=FLAGS.sheet_name, 
                                  label_name=FLAGS.label_name)

    n_features = X.shape[1]
    X, y = gradient_augment(X, y)
    X, y = temporalize(X, y, LOOKBACK)
    _, X_test, _, y_test = train_test_split(np.array(X), np.array(y), test_size=DATA_SPLIT_PCT, random_state=0)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], LOOKBACK, n_features)

    ## Loads scaler fit on training data.
    with open(FLAGS.data_scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    X_test_scaled = scale(X_test, scaler)

    model = keras.models.load_model(FLAGS.final_model_path)

    get_layer_output = tf.keras.backend.function([model.layers[0].input],
                                      [model.layers[1].output])
    layer_output = get_layer_output([X_test_scaled])[0]
    print(layer_output.shape, layer_output)

    classifier = LogisticRegression(class_weight='balanced', max_iter = 200, penalty = 'l1', solver = 'saga', C = 0.01, verbose=1)

    classifier.fit(layer_output[:,-384:], y_test)
    y_hat_test = classifier.predict(layer_output[:,-384:])
    print("Precision        Recall       F_score      Support")
    test_res = precision_recall_fscore_support(y_test, y_hat_test, average='binary')
    print(test_res)


if __name__ == '__main__':
    app.run(main)
