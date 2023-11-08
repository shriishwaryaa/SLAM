'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import matplotlib.pyplot as plt


def vectorize_state(traj, landmarks):
    x = np.concatenate((traj.flatten(), landmarks.flatten()))
    return x


def devectorize_state(x, n_poses):
    traj = x[:n_poses * 2].reshape((-1, 2))
    landmarks = x[n_poses * 2:].reshape((-1, 2))
    return traj, landmarks


def plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks, method=None, time=None):
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='gt poses')
    plt.scatter(gt_landmarks[:, 0],
                gt_landmarks[:, 1],
                c='b',
                marker='+',
                label='gt landmarks')

    plt.plot(traj[:, 0], traj[:, 1], 'r-', label='poses')
    plt.scatter(landmarks[:, 0],
                landmarks[:, 1],
                s=30,
                facecolors='none',
                edgecolors='r',
                label='landmarks')

    plt.legend()

    if time != None:
        img_name = f'../data/{method}.png'
        plt.title(f'Method: {method}, Average time: {time}')
    else:
        img_name = f'../data/nonlinear_{method}.png'
        plt.title(f'Method: {method}')
    plt.savefig(img_name, bbox_inches='tight', pad_inches=0.1)

    plt.show()
