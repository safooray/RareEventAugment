import numpy as np


"""## Data augmentation by adding noise."""
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
