import numpy as np
import scipy.io as scio
import os
import cv2

# 图片矢量化,读取图片，展平[h, w] -> [1, h*w]，并返回nparray
from numpy.random import permutation
np.random.seed(123)

def img2vector(image):
    img = cv2.imread(image, 0)
    rows, cols = img.shape
    imgVector = np.zeros((1, rows * cols))
    imgVector = np.reshape(img, (1, rows * cols))
    return imgVector


# 在此处修改数据集路径
IMG_PATH = "./data/Face Recognition Data/faces94/all/"  # face94原始数据是分成female、male、staff的，这里把他们事先copy到了一起
# IMG_PATH = "./data/Face Recognition Data/faces95/"
# IMG_PATH = "./data/Face Recognition Data/faces96/"
# IMG_PATH = "./data/Face Recognition Data/grimace/"

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
min_train = 5   # 用于训练的最少样本个数
max_train = 15   # 用于训练的最多样本个数
# 测试=总数-训练


# 读入人脸库,每个人随机选择k张作为训练集,其余构成测试集
def load_FRD(k):
    """
    对训练数据集进行数组初始化
    """
    train_face = np.zeros((N_PEOPLE * k, width * height))
    train_label = np.zeros(N_PEOPLE * k)
    test_face = np.zeros((N_PEOPLE * (IMG_PER_PEOPLE - k), width * height))
    test_label = np.zeros(N_PEOPLE * (IMG_PER_PEOPLE - k))

    sample = permutation(IMG_PER_PEOPLE) + 1  # 随机排序1-20 (0-19）+1
    for i in range(N_PEOPLE):  # 第i个人
        for j in range(IMG_PER_PEOPLE):  # 第j张照片
            img_file = os.path.join(IMG_PATH, str(people_name[i]), str(people_name[i])+'.'+str(sample[j]) + '.jpg')
            # 读取图片并展平成vector
            if os.path.isfile(img_file):
                img_data = img2vector(img_file)
                if j < k:
                    # 构成训练集
                    train_face[i * k + j, :] = img_data             # 人数*训练集人数+第几个图片
                    train_label[i * k + j] = i + 1
                else:
                    # 构成测试集
                    test_face[i * (IMG_PER_PEOPLE - k) + (j - k), :] = img_data
                    test_label[i * (IMG_PER_PEOPLE - k) + (j - k)] = i + 1

    return train_face, train_label, test_face, test_label, width,  height

# 加载PIE数据集，训练集太大，无法直接做PCA，这里直接随机抽了一部分样本，ratio表示抽取的比例
def load_PIE(ratio=0.5):
    train_face = np.empty((0, 4096))
    train_label = np.empty((0, 1))
    test_face = np.empty((0, 4096))
    test_label = np.empty((0, 1))
    # 把pose_xx都合并到一个array中
    for mat in os.listdir('./data/PIE dataset/'):
        mat = os.path.join('./data/PIE dataset/', mat)
        data_dict = scio.loadmat(mat)
        train_indices = np.squeeze(data_dict['isTest'] == 0)
        test_indices = np.squeeze(data_dict['isTest'] == 1)

        train_face = np.concatenate((train_face, data_dict['fea'][train_indices, :]))
        train_label = np.concatenate((train_label, data_dict['gnd'][train_indices, :]))
        test_face = np.concatenate((test_face, data_dict['fea'][test_indices, :]))
        test_label = np.concatenate((test_label, data_dict['gnd'][test_indices, :]))

    train_size = train_face.shape[0]
    idx = np.random.permutation(train_size)
    train_face = train_face[idx[:round(train_size * ratio)], :]
    train_label = train_label[idx[:round(train_size * ratio)], :]
    return train_face, train_label, test_face, test_label, 64, 64

