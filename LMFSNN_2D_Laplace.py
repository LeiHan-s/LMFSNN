import torch
import math
import time
import scipy
import datetime
import os
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.spatial import KDTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
import glob
import random
from matplotlib.colors import Normalize
from scipy.interpolate import griddata


def get_points_matlab(Nb2, Nii, Nb1):
    # 开启 MATLAB 引擎
    eng = matlab.engine.start_matlab()
    # 调用 MATLAB 函数
    data = eng.generateData(float(Nb2), float(Nii), float(Nb1))
    data = torch.tensor(data, dtype=torch.float64)
    # 关闭 MATLAB 引擎
    eng.quit()
    return data

def show_u_error_pred_solved(u_pred, u_solved):
    from matplotlib.colors import Normalize
    from scipy.interpolate import griddata

    folder_path = "U_fig"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    u_in, u_in_MFS = func(data).detach().cpu(),func(data_LMFS).detach().cpu()
    test_x,test_x_MFS = data.detach().cpu(), data_LMFS.detach().cpu()
    x_plt, y_plt = test_x[:,0], test_x[:,1]
    x_plt_MFS, y_plt_MFS = test_x_MFS[:, 0], test_x_MFS[:, 1]
    u_pred = u_pred.detach().cpu()
    u_solved = u_solved.detach().cpu()
    errors = torch.abs(u_pred.flatten() - u_in.flatten().squeeze())
    errors_solved = torch.abs(u_solved.flatten() - u_in_MFS.flatten().squeeze())

    def plot_interpolated_heatmap(X_grid, Y_grid, Z, title, norm=None):
        """
        根据输入的X_grid, Y_grid和Z数据进行插值，并在极坐标函数定义的边界内绘制热力图。

        参数:
        X_grid (numpy.ndarray): 不均匀分布的X坐标数据。
        Y_grid (numpy.ndarray): 不均匀分布的Y坐标数据。
        Z (numpy.ndarray): 对应的Z值。
        title (str): 图像的标题。
        """
        save_path = folder_path+'/'+title
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 定义均匀的网格
        x_uniform = np.linspace(-2, 2, 800)
        y_uniform = np.linspace(-2, 2, 800)
        X_uniform, Y_uniform = np.meshgrid(x_uniform, y_uniform)

        # 使用 griddata 进行插值
        Z_uniform = griddata((X_grid, Y_grid), Z, (X_uniform, Y_uniform), method='cubic')

        # 创建掩码：掩码区域为极坐标函数定义的边界外部
        r = lambda t: 1 + np.cos(2 * t) ** 2
        mask = np.sqrt(X_uniform ** 2 + Y_uniform ** 2) > r(np.arctan2(Y_uniform, X_uniform))

        # 应用掩码，将边界外部区域设置为 NaN
        Z_uniform[mask] = np.nan

        # 绘制热力图
        fig, ax = plt.subplots()
        cmap = plt.cm.viridis
        cax = ax.imshow(Z_uniform, cmap='jet', extent=[-2, 2, -2, 2], origin='lower', interpolation='bilinear', norm=norm)
        fig.colorbar(cax, ax=ax, orientation='vertical')


        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # 显示图像
        plt.savefig(save_path+'/u_solved_' + name_str + '.png')
        # plt.show()

    z_min = min(errors.min(), errors_solved.min())
    z_max = max(errors.max(), errors_solved.max())
    norm = Normalize(vmin=z_min, vmax=z_max)

    plot_interpolated_heatmap(x_plt, y_plt, u_pred, 'LMFSNN')
    plot_interpolated_heatmap(x_plt_MFS, y_plt_MFS, u_solved, 'LMFS')
    plot_interpolated_heatmap(x_plt, y_plt, errors, 'LMFSNN error')
    plot_interpolated_heatmap(x_plt_MFS, y_plt_MFS, errors_solved, 'LMFS error')

def plot_loss(loss_f, loss_b):
    folder_path = "loss"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    epochs = range(1, len(loss_f) + 1)
    epochs = [i*50 for i in epochs]
    # 绘制曲线图
    plt.plot(epochs, loss_f, 'g', label='loss_f')
    plt.plot(epochs, loss_b, 'b', label='loss_b')
    plt.plot(epochs, [i * 100 + j for i, j in zip(loss_b, loss_f)], 'r', label='loss')

    # 添加图例和标签
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')

    # 显示图形
    plt.savefig(folder_path+'/loss_' + name_str + '.png')
    plt.show()

# 定义保存检查点的方法
def save_checkpoint(model, folder='model'):
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 生成时间戳作为文件名的一部分
    timestamp = datetime.datetime.now().strftime('%m%d%H')
    filename = f'{folder}/model_{timestamp}_R{R}_n{Nb1 + Nb2 + Nii}_Nl{Nl}_Ns{Ns}.pth'

    # 保存模型的状态字典、自定义属性和模型参数
    torch.save({
        'model_state_dict': model.state_dict(),
        'custom_attributes_maxdm': {'maxdm': model.maxdm},
        'custom_attributes_rxy': {'rxy': model.rxy},
        'parameters': {  # 保存模型参数
            'Nii': Nii,
            'Nb1': Nb1,
            'Nb2': Nb2,
            'R': R,
            'Nl': Nl,
            'Ns': Ns
        }
    }, filename)

    print(f"Model saved as {filename}")

# 定义加载检查点的方法
def load_checkpoint(folder='model'):
    # 获取文件夹中所有模型文件
    model_files = sorted(glob.glob(os.path.join(folder, 'model_*.pth')), key=os.path.getmtime)

    if not model_files:
        raise FileNotFoundError(f"No model files found in {folder}")

    # 加载最新的模型文件
    latest_model_file = model_files[-1]
    checkpoint = torch.load(latest_model_file)

    # 加载保存的参数
    loaded_params = checkpoint['parameters']
    Nii = loaded_params['Nii']
    Nb1 = loaded_params['Nb1']
    Nb2 = loaded_params['Nb2']
    R = loaded_params['R']
    Nl = loaded_params['Nl']
    Ns = loaded_params['Ns']

    # 重新初始化模型并加载状态字典
    model = LMFSNN(Nl, R, Ns, {data_in, data_bound}).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.maxdm = checkpoint['custom_attributes_maxdm']['maxdm']
    model.rxy = checkpoint['custom_attributes_rxy']['rxy']

    print(f"Model loaded from {latest_model_file}")
    print(f"Loaded parameters: Nii={Nii}, Nb1={Nb1}, Nb2={Nb2}, R={R}, Nl={Nl}, Ns={Ns}")

    return model

class LMFSNN(torch.nn.Module):
    def __init__(self, Nl, R, Ns, data):
        super(LMFSNN, self).__init__()
        data_in, data_bound = data
        self.centers = torch.cat((data_in, data_bound))
        self.len_data = len(data_in)
        self.Nl = Nl
        self.R_init = R
        self.hidden_dim = len(self.centers)
        self.u = torch.nn.Parameter(torch.ones((self.hidden_dim, 1), dtype=torch.float64))
        self.R = torch.nn.Parameter(torch.ones((self.hidden_dim, 1), dtype=torch.float64) * R)
        self.Ns = Ns
        self.f_A = None
        self.tpx = None
        self.sinta = None
        self.maxdm = None
        self.i = 0
        self.RR = torch.ones((self.hidden_dim, 1), dtype=torch.float64, device=device) * R

        torch.nn.init.normal_(self.u, mean=0.0, std=0.01)

    def local_activate(self, x):
        x0 = np.array(self.centers.cpu())
        x = x.cpu().detach().numpy()
        kdtree = KDTree(x0)

        v1 = []  # collocation points
        v2 = []  # center points
        for i in range(len(x)):
            query_point = x[i]
            _, indices = kdtree.query(query_point, k=self.Nl + 1)  # 加1是因为最近邻中包括自身节点
            indices = indices[1:]
            indices.sort()
            v1.extend([i] * self.Nl)  # 将自身节点的索引重复k+1次添加到self_nodes列表中
            v2.extend(indices)  # 添加最近邻节点的索引到nearest_nodes列表中

        local = torch.cat((torch.tensor(v1).view(-1, 1), torch.tensor(v2).view(-1, 1)), dim=1).view(-1, self.Nl, 2)

        A_index_all = local
        A_inv_index_all = torch.zeros((len(x), self.Nl, self.Nl, 2), dtype=torch.int)
        A_inv_index_all[:, :, :, 1] = A_index_all[:, :, 1].unsqueeze(1).expand((len(x), self.Nl, self.Nl))
        A_inv_index_all[:, :, :, 0] = A_index_all[:, :, 1].unsqueeze(2).expand((len(x), self.Nl, self.Nl))

        self.A_index_all = A_index_all.long()
        self.A_inv_index_all = A_inv_index_all.long()


    def pinv(self, a, atol=1e-8):
        """
        计算矩阵的伪逆。
        参数：
            a (tensor): 要计算伪逆的矩阵，形状为(M, N)。
            atol (float): 绝对阈值项，可选参数，默认值为1e-8。
        返回值：
            伪逆矩阵B，形状为(N, M)。
        """
        u, s, vh = torch.svd(a)
        maxS = torch.max(s, dim=-1)
        val = atol + maxS.values * torch.finfo(s.dtype).eps
        rank = torch.sum(s > val.unsqueeze(-1), dim=-1)

        rank_dict = {}
        # 遍历列表，构建字典
        for index, value in enumerate(rank.cpu().detach().tolist()):
            if value in rank_dict:
                rank_dict[value].append(index)
            else:
                rank_dict[value] = [index]

        B = torch.zeros((a.size(0), a.size(2), a.size(1)), device=a.device, dtype=a.dtype)
        for rank in rank_dict.keys():
            u1 = u[rank_dict[rank], :, :rank]
            s1 = s[rank_dict[rank], :rank]
            s1_inv = torch.where(s1 < atol, torch.zeros_like(s1), 1.0 / s1)
            # 对角化s1
            rows, cols = s1_inv.shape
            diagonal_s1_inv = torch.zeros(rows, cols, cols, dtype=s1_inv.dtype, device=s1_inv.device)
            diagonal_s1_inv.as_strided([rows, cols], [cols * cols, cols + 1]).copy_(s1_inv)
            B[rank_dict[rank]] = torch.matmul(torch.matmul(vh[rank_dict[rank], :, :rank], diagonal_s1_inv),
                                              u1.transpose(-1, -2))

        return B

    def get_dist(self, x):
        size = (x.size(0), self.centers.size(0), x.size(1))
        c = self.centers.unsqueeze(0).expand(size).to(device)
        self.tpx = c[self.A_index_all[:, :, 0], self.A_index_all[:, :, 1], :]
        self.sinta = torch.tensor([i / self.Ns for i in range(self.Ns)]).cuda().unsqueeze(0).expand(
            (x.size(0), self.Ns)) * 2 * torch.pi
        self.maxdm = torch.max(
            torch.sqrt((x[:len(self.A_index_all)].unsqueeze(1) - self.tpx).pow(2).sum(-1).unsqueeze(1)))
        self.rxy = torch.cat(((torch.cos(self.sinta)).unsqueeze(2), (torch.sin(self.sinta)).unsqueeze(2)), dim=-1)

    def get_A(self, x):
        if self.sinta == None:
            self.get_dist(x)
        A_index_all = self.A_index_all

        c_r = self.R.unsqueeze(1) * self.maxdm * self.rxy

        c = self.centers.unsqueeze(1).to(device) + c_r
        c = c[:len(A_index_all)]
        dm = torch.sqrt((x[:len(A_index_all)].unsqueeze(1) - c).pow(2).sum(-1).unsqueeze(1))
        dm1 = torch.sqrt(torch.transpose((self.tpx.unsqueeze(1) - c.unsqueeze(2)).pow(2).sum(-1), 1, 2))

        mfs = torch.log(dm)
        mfs1 = torch.log(dm1)
        mfs_pinv = self.pinv(mfs1, 1e-8)

        f_A = torch.eye(self.hidden_dim, dtype=torch.float64).cuda()
        f_A0 = torch.zeros((self.hidden_dim, self.hidden_dim), dtype=torch.float64).cuda()
        f_A0[A_index_all[:, :, 0], A_index_all[:, :, 1]] = -torch.matmul(mfs, mfs_pinv).squeeze()
        f_A = f_A + f_A0

        return f_A

    def forward(self, x, t):
        if t % 5 == 0:
            self.f_A = self.get_A(x)
            self.RR = self.R.data
            self.i += 1

        f = torch.matmul(self.f_A, self.u)

        return f.squeeze(), self.u.squeeze(), self.f_A.squeeze()


def func(x):
    return torch.sinh(x[..., 0]) * torch.cos(x[..., 1])
def f_func(x):
    return torch.zeros_like(x[..., 1])


Nii = 1600  # 内部点数量
Nb1 = 210
Nb2 = 200
R = 5       # 初始源点半径
Nl = 30
Ns = 15

epoch = 10001
turb_size = 0.1  # 扰动比例系数
lr_R = 0.001
lr_U = 0.01

data = get_points_matlab(Nb2, Nii, Nb1).to(device)
data_in, data_bound = data[:int(Nii + Nb2)], data[int(Nii + Nb2):]
data = torch.cat((data_in, data_bound))
u, u_bound = func(data), func(data_bound)
f = torch.cat((f_func(data_in), func(data_bound)))

data_LMFS_bound = torch.cat((data_in[:Nb2], data_bound))
data_LMFS_in = torch.cat((data_LMFS_bound, data_in[Nb2:]))
data_LMFS = torch.cat((data_LMFS_in, data_LMFS_bound))
u_LMFS, u_LMFS_bound = func(data_LMFS), func(data_LMFS_bound)
f_MFS = torch.cat((f_func(data_LMFS_in), func(data_LMFS_bound)))

model = LMFSNN(Nl, R, Ns, {data_in, data_bound}).to(device)
model.local_activate(data_in)  # 使用KD-tree抓取邻域点

model_MFS = LMFSNN(Nl, R, Ns, {data_LMFS_in, data_LMFS_bound}).to(device)
model_MFS.local_activate(data_LMFS_in)

print(model.state_dict())

learning_rates = {'R': lr_R, 'u': lr_U}
model_parameters_dict = [
    {'params': model.R, 'lr': learning_rates['R']},
    {'params': model.u, 'lr': learning_rates['u']}
]
optimizer = torch.optim.Adam(model_parameters_dict)

loss_func = torch.nn.MSELoss()

scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=800, verbose=True, min_lr=1e-8)
name_str = '{}_epoch{}_n{}_Ns{}_R{}_lr{}&{}'.format(datetime.datetime.now().strftime("%m%d"), epoch, Nb1+Nb2+Nii, Ns, R, learning_rates['R'], learning_rates['u'])


# 训练
def train(model, epoch, turb_size=0.1):
    model.local_activate(data_in)

    loss_f = []
    loss_b = []
    l1_u_list = []
    lmax_u_list = []
    best_loss = float('inf')

    _, _, fA_MFS = model_MFS(data_LMFS, 0)
    u_solved = torch.linalg.solve(fA_MFS, f_MFS)
    l1_u_solved = torch.nn.L1Loss()(u_solved, u_LMFS).item()
    lmax_u_solved = torch.max(torch.abs(u_solved - u_LMFS)).item()

    # 记录开始时间
    start_time = time.time()
    for t in range(epoch):
        optimizer.zero_grad()

        # 前向传播: 输入b, 输出预测值
        f_pred, u_pred, f_A = model(data, t)

        l_f = loss_func(f_pred, f)
        l_bound1 = loss_func(u_pred[-len(data_bound):], u_bound)
        l_bound2 = loss_func(u_pred[:Nb2], u[:Nb2])

        # 计算损失
        loss = l_f + 100 * (l_bound1 + l_bound2)

        # 反向传播和优化
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)  # 更新学习率


        if t % 50 == 0:
            loss_f.append(l_f.item())
            loss_b.append(l_bound1.item() + l_bound2.item())
            l1_u = torch.nn.L1Loss()(u_pred, u)
            lmax_u = torch.max(torch.abs(u_pred - u))
            l1_u_list.append(l1_u.item())
            lmax_u_list.append(lmax_u.item())

            print("Epoch {}, Loss: {}, L(f): {}, L(b1): {}, L(b2): {}"
                  .format(t,
                          loss.item(),
                          l_f.item(),
                          l_bound1.item(),
                          l_bound2.item()))
            print('mean_loss: {} ({})'.format(l1_u.item(), l1_u_solved))
            print('max_loss: {} ({})'.format(lmax_u.item(), lmax_u_solved), '\n')

            # 如果损失减少，则更新并保存模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(model)


        model.R.data = model.R.data + torch.randn_like(model.R, device=model.R.device, dtype=model.R.dtype) * \
                   optimizer.param_groups[0]['lr'] * turb_size
    # 打印训练时间
    print("模型训练时间：", time.time() - start_time, "秒", model.i)

    model = load_checkpoint()

    # 创建包含训练数据的字典
    variables_dict = {
        "loss_f": loss_f,
        "l1_u_list": l1_u_list,
        "lmax_u_list": lmax_u_list,
        'l1_u_solved': l1_u_solved,
        'lmax_u_solved': lmax_u_solved,
    }

    # 保存字典到指定文件
    if not os.path.exists('train_data'):
        os.makedirs('train_data')

    file_path = "train_data/" + name_str + '.json'
    with open(file_path, "w") as file:
        json.dump(variables_dict, file)

    return loss_f, loss_b, l1_u_list, lmax_u_list, u_pred, u_solved, model.R.data

# model = load_checkpoint()
loss_f, loss_b, l1_u_list, lmax_u_list, u_pred, u_solved, R = train(model, epoch, turb_size=turb_size)
plot_loss(loss_f, loss_b)
show_u_error_pred_solved(u_pred, u_solved)



