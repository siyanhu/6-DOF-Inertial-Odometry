import numpy as np
import pandas as pd
import quaternion
import scipy.interpolate
import torch
from torch.utils.data import Dataset, DataLoader

def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
    interpolated = func(output_timestamp)
    return interpolated

def load_euroc_mav_dataset(imu_data_filename, gt_data_filename):
    gt_data = pd.read_csv(gt_data_filename).values    
    imu_data = pd.read_csv(imu_data_filename).values

    gyro_data = interpolate_3dvector_linear(imu_data[:, 1:4], imu_data[:, 0], gt_data[:, 0])
    acc_data = interpolate_3dvector_linear(imu_data[:, 4:7], imu_data[:, 0], gt_data[:, 0])
    pos_data = gt_data[:, 1:4]
    ori_data = gt_data[:, 4:8]

    return gyro_data, acc_data, pos_data, ori_data

def load_oxiod_dataset(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values
    gt_data = pd.read_csv(gt_data_filename).values

    imu_data = imu_data[1200:-300]
    gt_data = gt_data[1200:-300]

    gyro_data = imu_data[:, 4:7]
    acc_data = imu_data[:, 10:13]
    
    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    return gyro_data, acc_data, pos_data, ori_data

def force_quaternion_uniqueness(q):
    q_data = quaternion.as_float_array(q)

    if np.absolute(q_data[0]) > 1e-05:
        return -q if q_data[0] < 0 else q
    elif np.absolute(q_data[1]) > 1e-05:
        return -q if q_data[1] < 0 else q
    elif np.absolute(q_data[2]) > 1e-05:
        return -q if q_data[2] < 0 else q
    else:
        return -q if q_data[3] < 0 else q

def cartesian_to_spherical_coordinates(point_cartesian):
    delta_l = np.linalg.norm(point_cartesian)

    if np.absolute(delta_l) > 1e-05:
        theta = np.arccos(point_cartesian[2] / delta_l)
        psi = np.arctan2(point_cartesian[1], point_cartesian[0])
        return delta_l, theta, psi
    else:
        return 0, 0, 0


class IMUDataset(Dataset):
    def __init__(self, gyro_data, acc_data, pos_data, ori_data, window_size=200, stride=10):
        self.gyro_data = torch.tensor(gyro_data, dtype=torch.float32)
        self.acc_data = torch.tensor(acc_data, dtype=torch.float32)
        self.pos_data = torch.tensor(pos_data, dtype=torch.float32)
        self.ori_data = torch.tensor(ori_data, dtype=torch.float32)
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return (self.gyro_data.shape[0] - self.window_size - 1) // self.stride

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size

        x_gyro = self.gyro_data[start_idx + 1 : end_idx + 1]
        x_acc = self.acc_data[start_idx + 1 : end_idx + 1]

        p_a = self.pos_data[start_idx + self.window_size//2 - self.stride//2]
        p_b = self.pos_data[start_idx + self.window_size//2 + self.stride//2]

        q_a = quaternion.from_float_array(self.ori_data[start_idx + self.window_size//2 - self.stride//2].numpy())
        q_b = quaternion.from_float_array(self.ori_data[start_idx + self.window_size//2 + self.stride//2].numpy())

        delta_p = torch.tensor(np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b - p_a).numpy()), dtype=torch.float32)
        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)
        delta_q = torch.tensor(quaternion.as_float_array(delta_q), dtype=torch.float32)

        return x_gyro, x_acc, delta_p, delta_q

def load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size=200, stride=10, batch_size=32):
    dataset = IMUDataset(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    init_p = torch.tensor(pos_data[window_size//2 - stride//2], dtype=torch.float32)
    init_q = torch.tensor(ori_data[window_size//2 - stride//2], dtype=torch.float32)

    return dataloader, init_p, init_q