import pandas as pd
import numpy as np
import quaternion
import matplotlib
from matplotlib import pyplot as plt


def force_quaternion_uniqueness(q):

    q_data = quaternion.as_float_array(q)

    if np.absolute(q_data[0]) > 1e-05:
        if q_data[0] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[1]) > 1e-05:
        if q_data[1] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[2]) > 1e-05:
        if q_data[2] < 0:
            return -q
        else:
            return q
    else:
        if q_data[3] < 0:
            return -q
        else:
            return q
        

def load_dataset_6d_quat(pos_data, ori_data):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    init_p = pos_data[0, :]
    init_q = ori_data[0, :]

    y_delta_p = []
    y_delta_q = []

    for idx in range(1, pos_data.shape[0]):

        p_a = pos_data[idx - 1, :]
        p_b = pos_data[idx, :]

        q_a = quaternion.from_float_array(ori_data[idx - 1, :])
        q_b = quaternion.from_float_array(ori_data[idx, :])

        delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))


    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))

    return [y_delta_p, y_delta_q], init_p, init_q


def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    cur_p = np.array(init_p)
    cur_q = quaternion.from_float_array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p, delta_q] in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + np.matmul(quaternion.as_rotation_matrix(cur_q), delta_p.T).T
        cur_q = cur_q * quaternion.from_float_array(delta_q).normalized()
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))


def load_gt_dataset(gt_data_filename):
    gt_data = pd.read_csv(gt_data_filename).values
    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    return pos_data, ori_data


gt_file = "/Users/siyanhu/Downloads/DEFAULT_DEFAULT_2024-08-14_13-32-04/Gt/vis.csv"

cur_pos_data, cur_ori_data = load_gt_dataset(gt_file)

[y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_pos_data, cur_ori_data)

gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)

matplotlib.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=[14.4, 10.8])
ax = fig.add_subplot(projection = '3d')
ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
min_x = gt_trajectory[:, 0]
min_y = gt_trajectory[:, 1]
min_z = gt_trajectory[:, 2]
max_x = gt_trajectory[:, 0]
max_y = gt_trajectory[:, 1]
max_z = gt_trajectory[:, 2]
range_x = np.absolute(max_x - min_x)
range_y = np.absolute(max_y - min_y)
range_z = np.absolute(max_z - min_z)
max_range = np.maximum(np.maximum(range_x, range_y), range_z)
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_zlim(-7, 7)
plt.show()
plt.clf()

fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot()
ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 2])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
plt.show()