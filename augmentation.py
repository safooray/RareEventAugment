import numpy as np


def noise_pos_augment(x, y):
    x_pos = x[y == 1]

    noise_matrix = np.random.normal(loc=0, scale=0.01, size=x_pos.shape)

    data_with_noise = x_pos + noise_matrix

    augmented_x = np.concatenate([x, data_with_noise], axis=0)
    augmented_y = np.concatenate([y, np.ones(shape=x_pos.shape[0])], axis=0)
    return augmented_x, augmented_y

def resample_augment(x, y):
    x_pos = x[y == 1]
    frac = x.shape[0] // x_pos.shape[0]
    resampled_data = []
    for i in range(frac):
        resampled_data.append(x_pos)
    resampled_data = np.concatenate(resampled_data, axis=0)
    augmented_x = np.concatenate([x, resampled_data], axis=0)
    augmented_y = np.concatenate([y, np.ones(shape=resampled_data.shape[0])], axis=0)
    return augmented_x, augmented_y

def gradient_augment(x, y, order=2):
    x_grad = np.gradient(x, axis=0)
    x_2nd_grad = np.gradient(x_grad, axis=0)
    if order == 1:
        augmented_x = np.concatenate([x, x_grad], axis=1)
    if order == 2:
        augmented_x = np.concatenate([x, x_grad, x_2nd_grad], axis=1)
    return augmented_x, y


def interpolation_augment(x, y):
    pass


from data_utils import prepare_data
if __name__ == '__main__':
    x, y = prepare_data()
    print(x.shape, y.shape)
    x, y = gradient_augment(x, y)
    print(x.shape, y.shape)
