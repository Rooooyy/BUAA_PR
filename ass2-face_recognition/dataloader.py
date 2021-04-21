import numpy as np
import scipy.io as scio
import os


def load_PIE():
    train_face = np.empty((0, 4096))
    train_label = np.empty((0, 1))
    test_face = np.empty((0, 4096))
    test_label = np.empty((0, 1))
    for mat in os.listdir('./data/PIE dataset/'):
        mat = os.path.join('./data/PIE dataset/', mat)
        data_dict = scio.loadmat(mat)
        train_indices = np.squeeze(data_dict['isTest'] == 0)
        test_indices = np.squeeze(data_dict['isTest'] == 1)

        train_face = np.concatenate((train_face, data_dict['fea'][train_indices, :]))
        train_label = np.concatenate((train_label, data_dict['gnd'][train_indices, :]))
        test_face = np.concatenate((test_face, data_dict['fea'][test_indices, :]))
        test_label = np.concatenate((test_label, data_dict['gnd'][test_indices, :]))

    return train_face, train_label, test_face, test_label

