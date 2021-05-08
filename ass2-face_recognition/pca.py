# coding:utf-8
import os
from numpy import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import orth
from pylab import mpl
from dataloader import load_PIE, load_FRD

mpl.rcParams['font.sans-serif'] = ['SimHei']

datasets = 'FRD'        # PIE or FRD

# 参数
feature_n = 101  # 需要的特征（最多）
min_feature = 10  # 需要的特征（最少）
gap = 10    # 最少到最多的间隔
min_train = 5   # 用于训练的最少样本个数
max_train = 15   # 用于训练的最多样本个数
# 测试=总数-训练

# 定义PCA算法
def PCA(data, r):
    data = np.float32(np.mat(data))  # 用mat会比ndarray计算快一些       (1824, 38416) 训练集个数152个人*每个人选择12个，原始图片特征维度
    rows, cols = np.shape(data)
    data_mean = np.mean(data, 0)  # 对列求平均值
    A = data - np.tile(data_mean, (rows, 1))  # 将所有样例减去对应均值得到A
   
    # ATA是协方差矩阵
    # ATA和AAT有相同的 非零 特征值，但AAT比ATA规模小很多，可以简化计算
    # AATα = λα
    # ATA(ATα) = λ(ATα)
    # 所以ATα 是 ATA的特诊值
    C = A * A.T  # np.mat可以直接用*作为矩阵乘法
    D, V = np.linalg.eig(C)  # 求协方差矩阵的特征值和特征向量

    # 逆序排序
    indices = argsort(D, )[::-1]  # eig返回的特征值并不是排序的，所以要排序后再选择前r个主成份

    # 选择前r个
    V_r = V[:, indices[:r]]

    # 贡献率
    sum=0
    for i in range(r):
        sum += D[indices[i]]
    print('当选择%d个主成分时，其贡献率为%.3f' %(r, sum/D.sum()))

    V_r = A.T * V_r  # A.T*V_r是ATA的特征向量

    V_r = orth(V_r)  # 用scipy.linalg.orth求单位正交向量，orth是用svd做内核的，比直接用np.linalg.eig(ATA)快

    final_data = A * V_r
    return final_data, data_mean, V_r

# 人脸识别
def face_rec(datasets='PIE'):

    for r in range(min_feature, feature_n, gap):                        # 遍历使用特征数不同时，精度差异
        print("当选择%d个主成分时" % r)
        x_value = []
        y_value = []
        # 这里是循环多少个图片作为训练集
        # for k in range(min_train, max_train + 1):
        for k in range(15,16):
            # 加载数据集, train_size=k test_size = IMG_PER_PEOPLE - k
            if datasets=='FRD':
                train_face, train_label, test_face, test_label, width, height = load_FRD(k=k)           # 20个中选择k个作为训练集
            elif datasets=='PIE':
                train_face, train_label, test_face, test_label, width, height = load_PIE(ratio=0.5)

            # 利用PCA算法进行训练
            data_train_new, data_mean, V_r = PCA(train_face, r)             # 将训练集样本全部投影到低维空间，平均脸是所有训练样本的平均，V_r是投影向量

            num_train = data_train_new.shape[0]  # 训练脸总数
            num_test = test_face.shape[0]        # 测试脸总数

            temp_face = test_face - np.tile(data_mean, (num_test, 1))   # 中心化，因为训练数据在PCA时也进行了中心化
            data_test_new = temp_face * V_r                             # 把test_face在同一组基下进行投影  (num, features)
            # mat to array
            data_test_new = np.array(data_test_new)
            data_train_new = np.array(data_train_new)

            # 测试准确度
            true_num = 0
            for i in range(num_test):
                test_sample = data_test_new[i, :]           # (features)
                diffMat = data_train_new - np.tile(test_sample, (num_train, 1))  # 训练数据与测试脸之间距离
                sqDiffMat = diffMat ** 2                    # 找出当前测试样本  和 全部训练集中哪个样本最像
                sqDistances = sqDiffMat.sum(axis=1)  # 按行求和
                sortedDistIndices = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
                indexMin = sortedDistIndices[0]  # 距离最近的索引
                if train_label[indexMin] == test_label[i]:
                    true_num += 1
                else:
                    pass

            accuracy = float(true_num) / num_test
            x_value.append(k)
            y_value.append(round(accuracy, 2))

            print('当每个人选择%d张照片进行训练时，准确率为: %.2f%%' % (train_face.shape[0], accuracy * 100))
            print('训练集为%d, 测试集为%d，准确率为: %.2f%%' % (train_face.shape[0], test_face.shape[0], accuracy * 100))

        # 相同的数据集 特征脸 和 平均脸是一样的
        # 显示平均脸
        plt.imshow(np.array(data_mean.reshape(height, width)), cmap ='gray')
        plt.show()

        # 显示特征脸
        plt.figure()
        for i in range(10):
            plt.subplot(2, 5, i+1)
            title="Eigenface"+str(i+1)
            #行，列，索引
            plt.imshow(np.real(V_r[:, i].reshape(height, width)), cmap ='gray')
            plt.title(title, fontsize=8)
            plt.xticks([])
            plt.yticks([])
        plt.show()

if __name__ == '__main__':
    face_rec(datasets)
