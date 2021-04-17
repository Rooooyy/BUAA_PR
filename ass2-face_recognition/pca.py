# coding:utf-8
import os
from numpy import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


# 图片矢量化,读取图片，展平[h, w] -> [1, h*w]，并返回nparray  
def img2vector(image):
    img = cv2.imread(image, 0)
    rows, cols = img.shape
    imgVector = np.zeros((1, rows * cols))
    imgVector = np.reshape(img, (1, rows * cols))
    return imgVector


# 在此处修改数据集路径
# IMG_PATH = "./data/Face Recognition Data/faces94/all/"  # face94原始数据是分成female、male、staff的，这里把他们事先copy到了一起
# IMG_PATH = "./data/Face Recognition Data/faces95/"
# IMG_PATH = "./data/Face Recognition Data/faces96/"
IMG_PATH = "./data/Face Recognition Data/grimace/"  # 准确率都是100%，有问题

# faces94、faces95、grimace的数据都是180*200, face96是196*196
if IMG_PATH.split('/')[2] == "Face Recognition Data":
    if IMG_PATH.split('/')[3] == "faces96":
        width = 196
        height = 196
    else:
        width = 180     
        height = 200


# 读取数据中的所有人名，也就是文件夹的名称  
people_name = []
for filename in os.listdir(IMG_PATH):
    pathname = os.path.join(IMG_PATH, filename)
    if os.path.isfile(filename):
        continue
    else:
        people_name.append(filename)

N_PEOPLE = len(people_name)  # 总人数
IMG_PER_PEOPLE = 20  # 对每个人都抽取IMG_PER_PEOPLE张照片作为数据集
# 参数
feature_n = 41  # 需要的特征（最多）
min_feature = 10  # 需要的特征（最少）
gap = 10    # 最少到最多的间隔
min_train = 5   # 用于训练的最少样本个数
max_train = 15   # 用于训练的最多样本个数
# 测试=总数-训练


# 读入人脸库,每个人随机选择k张作为训练集,其余构成测试集
def load_data(k):
    """
    对训练数据集进行数组初始化
    """
    train_face = np.zeros((N_PEOPLE * k, width * height))
    train_label = np.zeros(N_PEOPLE * k)
    test_face = np.zeros((N_PEOPLE * (IMG_PER_PEOPLE - k), width * height))
    test_label = np.zeros(N_PEOPLE * (IMG_PER_PEOPLE - k))

    sample = random.permutation(IMG_PER_PEOPLE) + 1  # 随机排序1-20 (0-19）+1
    for i in range(N_PEOPLE):  # 第i个人
        for j in range(IMG_PER_PEOPLE):  # 第j张照片
            # image = IMG_PATH + '/' + str(people_name[i]) + '/' + str(people_name[i])+'.'+str(sample[j]) + '.jpg'
            img_file = os.path.join(IMG_PATH, str(people_name[i]), str(people_name[i])+'.'+str(sample[j]) + '.jpg')
            # 读取图片并展平成vector
            if os.path.isfile(img_file):
                img_data = img2vector(img_file)
                if j < k:
                    # 构成训练集
                    train_face[i * k + j, :] = img_data
                    train_label[i * k + j] = i + 1
                else:
                    # 构成测试集
                    test_face[i * (IMG_PER_PEOPLE - k) + (j - k), :] = img_data
                    test_label[i * (IMG_PER_PEOPLE - k) + (j - k)] = i + 1

    return train_face, train_label, test_face, test_label


# 定义PCA算法
def PCA(data, r):
    data = np.float32(np.mat(data))
    rows, cols = np.shape(data)
    data_mean = np.mean(data, 0)  # 对列求平均值
    A = data - np.tile(data_mean, (rows, 1))  # 将所有样例减去对应均值得到A
    C = A * A.T  # 得到协方差矩阵
    D, V = np.linalg.eig(C)  # 求协方差矩阵的特征值和特征向量
    V_r = V[:, 0:r]  # 按列取前r个特征向量
    V_r = A.T * V_r  # 小矩阵特征向量向大矩阵特征向量过渡
    for i in range(r):
        V_r[:, i] = V_r[:, i] / (np.linalg.norm(V_r[:, i]) + 1e-5)  # 特征向量归一化, 防止出现除0的情况，加上一个epsilon

    final_data = A * V_r
    return final_data, data_mean, V_r


# 人脸识别
def face_rec():
    for r in range(min_feature, feature_n, gap):
        print("当选择%d个主成分时" % r)
        x_value = []
        y_value = []
        for k in range(min_train, max_train + 1):
            train_face, train_label, test_face, test_label = load_data(k)  # 得到数据集

            # 利用PCA算法进行训练
            data_train_new, data_mean, V_r = PCA(train_face, r)
            num_train = data_train_new.shape[0]  # 训练脸总数
            num_test = test_face.shape[0]  # 测试脸总数
            temp_face = test_face - np.tile(data_mean, (num_test, 1))
            data_test_new = temp_face * V_r  # 得到测试脸在特征向量下的数据
            data_test_new = np.array(data_test_new)  # mat change to array
            data_train_new = np.array(data_train_new)

            # 测试准确度
            true_num = 0
            for i in range(num_test):
                testFace = data_test_new[i, :]
                diffMat = data_train_new - np.tile(testFace, (num_train, 1))  # 训练数据与测试脸之间距离
                sqDiffMat = diffMat ** 2
                sqDistances = sqDiffMat.sum(axis=1)  # 按行求和
                sortedDistIndicies = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
                indexMin = sortedDistIndicies[0]  # 距离最近的索引
                if train_label[indexMin] == test_label[i]:
                    true_num += 1
                else:
                    pass

            accuracy = float(true_num) / num_test
            x_value.append(k)
            y_value.append(round(accuracy, 2))

            print('当每个人选择%d张照片进行训练时，The classify accuracy is: %.2f%%' % (k, accuracy * 100))
        '''
        # 绘图
        if r == 10:
            y1_value = y_value
            plt.plot(x_value, y_value, marker="o", markerfacecolor="red")
            for a, b in zip(x_value, y_value):
                plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)

            plt.title(u"降到10维时识别准确率", fontsize=14)
            plt.xlabel(u"K值", fontsize=14)
            plt.ylabel(u"准确率", fontsize=14)
            plt.show()
            # print ('y1_value',y1_value)
        if r == 20:
            y2_value = y_value
            plt.plot(x_value, y2_value, marker="o", markerfacecolor="red")
            for a, b in zip(x_value, y_value):
                plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)

            plt.title(u"降到20维时识别准确率", fontsize=14)
            plt.xlabel(u"K值", fontsize=14)
            plt.ylabel(u"准确率", fontsize=14)
            plt.show()
            # print ('y2_value',y2_value)
        if r == 30:
            y3_value = y_value
            plt.plot(x_value, y3_value, marker="o", markerfacecolor="red")
            for a, b in zip(x_value, y_value):
                plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)

            plt.title(u"降到30维时识别准确率", fontsize=14)
            plt.xlabel(u"K值", fontsize=14)
            plt.ylabel(u"准确率", fontsize=14)
            plt.show()
            # print ('y3_value',y3_value)
        if r == 40:
            y4_value = y_value
            plt.plot(x_value, y4_value, marker="o", markerfacecolor="red")
            for a, b in zip(x_value, y_value):
                plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)

            plt.title(u"降到40维时识别准确率", fontsize=14)
            plt.xlabel(u"K值", fontsize=14)
            plt.ylabel(u"准确率", fontsize=14)
            plt.show()
            # print ('y4_value',y4_value)

    # 各维度下准确度比较
    L1, = plt.plot(x_value, y1_value, marker="o", markerfacecolor="red")
    L2, = plt.plot(x_value, y2_value, marker="o", markerfacecolor="red")
    L3, = plt.plot(x_value, y3_value, marker="o", markerfacecolor="red")
    L4, = plt.plot(x_value, y4_value, marker="o", markerfacecolor="red")
    # for a, b in zip(x_value, y1_value):
    #    plt.text(a,b,(a,b),ha='center', va='bottom', fontsize=10)

    plt.legend([L1, L2, L3, L4], [u"降到10维", u"降到20维", u"降到30维", u"降到40维"], loc=4)
    plt.title(u"各维度识别准确率比较", fontsize=14)
    plt.xlabel(u"K值", fontsize=14)
    plt.ylabel(u"准确率", fontsize=14)
    plt.show()
    '''


if __name__ == '__main__':
    face_rec()
