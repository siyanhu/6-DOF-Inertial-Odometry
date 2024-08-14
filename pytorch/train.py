import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import shuffle

from dataset import load_oxiod_dataset, load_euroc_mav_dataset, load_dataset_6d_quat, IMUDataset
from model import create_model_6d_quat, CustomMultiLossLayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oxiod', 'euroc'], help='Training dataset name (\'oxiod\' or \'euroc\')')
    parser.add_argument('output', help='Model output name')
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    window_size = 200
    stride = 10

    imu_data_filenames = []
    gt_data_filenames = []

    if args.dataset == 'oxiod':
        # Add Oxford Inertial Odometry Dataset filenames here
        pass
    elif args.dataset == 'euroc':
        # Add EuRoC MAV Dataset filenames here
        pass

    datasets = []

    for cur_imu_data_filename, cur_gt_data_filename in zip(imu_data_filenames, gt_data_filenames):
        if args.dataset == 'oxiod':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)
        elif args.dataset == 'euroc':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_euroc_mav_dataset(cur_imu_data_filename, cur_gt_data_filename)

        [cur_x_gyro, cur_x_acc], [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data, window_size, stride)

        datasets.append(IMUDataset(cur_x_gyro, cur_x_acc, cur_y_delta_p, cur_y_delta_q))

    combined_dataset = ConcatDataset(datasets)
    train_size = int(0.9 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model_6d_quat(window_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = CustomMultiLossLayer()

    writer = SummaryWriter()

    best_val_loss = float('inf')
    for epoch in range(500):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_gyro, x_acc, y_delta_p, y_delta_q = [b.to(device) for b in batch]
            optimizer.zero_grad()
            pred_delta_p, pred_delta_q = model(x_gyro, x_acc)
            loss = criterion(pred_delta_p, pred_delta_q, y_delta_p, y_delta_q)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_gyro, x_acc, y_delta_p, y_delta_q = [b.to(device) for b in batch]
                pred_delta_p, pred_delta_q = model(x_gyro, x_acc)
                loss = criterion(pred_delta_p, pred_delta_q, y_delta_p, y_delta_q)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{args.output}_best.pth')

    torch.save(model.state_dict(), f'{args.output}_final.pth')
    writer.close()

    # Plot loss curves
    plt.figure()
    plt.plot(writer.get_scalar('Loss/train'), label='Train')
    plt.plot(writer.get_scalar('Loss/val'), label='Validation')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()