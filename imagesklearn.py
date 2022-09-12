# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import matplotlib.pyplot
import numpy
import scipy
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy
import PIL.Image


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import os
import pandas as pd
from scipy.optimize import linear_sum_assignment

# task1

def moonswork():
    matplotlib.pyplot.figure("yichi")
    matplotlib.pyplot.subplot(1,2,1)
    moons_X, moons_Y = make_moons(n_samples=400, noise=0.1)
    matplotlib.pyplot.title("make_moon")
    matplotlib.pyplot.scatter(moons_X[:, 0], moons_X[:, 1], s=100, marker="o", edgecolors='m', c=moons_Y, cmap='viridis')

    # 以下是第三个project中添加的内容
    matplotlib.pyplot.subplot(1,2,2)
    kmeans = KMeans(n_clusters=2)
    # 两个聚类中心 ↑
    kmeans.fit(moons_X)
    kmeans_y = kmeans.predict(moons_X)
    matplotlib.pyplot.title("make_moon by kmeans")
    matplotlib.pyplot.scatter(moons_X[:, 0], moons_X[:, 1], s=100, marker="o", edgecolors='m', c=kmeans_y, cmap='viridis')
    # 三大指标
    acc = accuracy_score(moons_Y, kmeans_y)
    nmi = normalized_mutual_info_score(moons_Y, kmeans_y)
    ari = adjusted_rand_score(moons_Y, kmeans_y)
    print("ACC=", acc)
    print("NMI=", nmi)
    print("ARI=", ari)
    print(

    )
    matplotlib.pyplot.show()

moonswork()

# task2

def circlework():
    matplotlib.pyplot.figure("ni")
    matplotlib.pyplot.subplot(1,2,1)
    circle_X,circle_Y = make_circles(n_samples=400, noise=0.1, factor=0.1)
    matplotlib.pyplot.title("make_circle")
    matplotlib.pyplot.scatter(circle_X[:, 0], circle_X[:, 1], s=100, marker="o", edgecolors='m', c=circle_Y, cmap='viridis')

    # 以下是第三个project中添加的内容
    matplotlib.pyplot.subplot(1,2,2)
    kmeans = KMeans(n_clusters=2)
    # 两个聚类中心 ↑
    kmeans.fit(circle_X)
    kmeans_y = kmeans.predict(circle_X)
    matplotlib.pyplot.title("make_circle by kmeans")
    matplotlib.pyplot.scatter(circle_X[:, 0], circle_X[:, 1], s=100, marker="o", edgecolors='m', c=kmeans_y, cmap='viridis')
    # 三大指标
    acc = accuracy_score(circle_Y, kmeans_y)
    nmi = normalized_mutual_info_score(circle_Y, kmeans_y)
    ari = adjusted_rand_score(circle_Y, kmeans_y)

    print("ACC=", acc)
    print("NMI=", nmi)
    print("ARI=", ari)



    matplotlib.pyplot.show()

circlework()

# task3

def blobswork():
    matplotlib.pyplot.figure("sany")
    matplotlib.pyplot.subplot(1,2,1)
    blob_X,blob_Y = make_blobs(n_samples=400, centers=3)
    matplotlib.pyplot.title("make_blobs")
    matplotlib.pyplot.scatter(blob_X[:, 0], blob_X[:, 1], s=100, marker="o", edgecolors='m', c=blob_Y, cmap='viridis')

    # 以下是第三个project中添加的内容
    matplotlib.pyplot.subplot(1,2,2)
    kmeans = KMeans(n_clusters=3)
    # 两个聚类中心 ↑
    kmeans.fit(blob_X)
    kmeans_y = kmeans.predict(blob_X)
    matplotlib.pyplot.title("make_blobs by kmeans")
    matplotlib.pyplot.scatter(blob_X[:, 0], blob_X[:, 1], s=100, marker="o", edgecolors='m', c=kmeans_y, cmap='viridis')
    # 三大指标
    acc = accuracy_score(blob_Y, kmeans_y)
    nmi = normalized_mutual_info_score(blob_Y, kmeans_y)
    ari = adjusted_rand_score(blob_Y, kmeans_y)

    print("ACC=", acc)
    print("NMI=", nmi)
    print("ARI=", ari)



    matplotlib.pyplot.show()

blobswork()


# task4

paths = "E:\\test\\picture.jpg"
img = plt.imread(paths)

x = numpy.array(img)
row, col, dim =x.shape
# 少一个变量不行？？？？没写dim卡了好久
x_train = x.reshape(-1,3)
print(x_train.shape)
print(x_train.size)
print(row, col, dim)

plt.figure("task4")
plt.subplot(2, 3, 1)
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.title("Picture")

# 第一张是原图
plt.imshow(img)

# 第二到第六张图
for t in range(2, 7):
    index = '23' + str(t)
    plt.subplot(int(index))
    tmp = KMeans(n_clusters=t).fit_predict(x_train)

    tmp = tmp.reshape(row, col)

    newpic = PIL.Image.new("RGB", (row, col))
    for i in range(row):
        for j in range(col):
            if tmp[i][j] == 0:
                newpic.putpixel((i, j), (0, 0, 255))
            elif tmp[i][j] == 1:
                newpic.putpixel((i, j), (255, 0, 0))
            elif tmp[i][j] == 2:
                newpic.putpixel((i, j), (0, 255, 0))
            elif tmp[i][j] == 3:
                newpic.putpixel((i, j), (77, 0, 200))
            elif tmp[i][j] == 4:
                newpic.putpixel((i, j), (177, 255, 77))
            elif tmp[i][j] == 5:
                newpic.putpixel((i, j), (177, 255, 177))
            elif tmp[i][j] == 6:
                newpic.putpixel((i, j), (177, 77, 255))
    title = "k=" + str(t)
    plt.title(title)
    plt.imshow(newpic)
    plt.axis('off')
plt.show()

# task5

def getinfo():
    # 获取文件并构成向量
    # 预测值为1维，把一张图片的三维压成1维，那么n张图片就是二维
    global total_photo
    file = os.listdir(r'E:\test\face_images\\')
    i = 0
    for subfile in file:
        photo = os.listdir(r'E:\test\face_images\\' + subfile)  # 文件路径自己改
        for name in photo:
            photo_name.append(r'E:\test\face_images\\' + subfile + '\\' + name)
            target.append(i)
        i += 1
    for path in photo_name:
        photo = imgplt.imread(path)
        photo = photo.reshape(1, -1)
        photo = pd.DataFrame(photo)
        total_photo = total_photo.append(photo, ignore_index=True)
    total_photo = total_photo.values


def kmeans():
    clf = KMeans(n_clusters=10)
    clf.fit(total_photo)
    y_predict = clf.predict(total_photo)
    centers = clf.cluster_centers_
    result = centers[y_predict]
    result = result.astype("int64")
    result = result.reshape(200, 200, 180, 3)  # 图像的矩阵大小为200,180,3
    return result, y_predict


def draw():
    fig, ax = plt.subplots(nrows=10, ncols=20, sharex=True, sharey=True, figsize=[15, 8], dpi=80)
    plt.subplots_adjust(wspace=0, hspace=0)
    count = 0
    for i in range(10):
        for j in range(20):
            ax[i, j].imshow(result[count])
            count += 1
    plt.xticks([])
    plt.yticks([])
    plt.show()


def score():
    ACC = accuracy_score(target, y_predict)  # y 真实值 y_predict 预测值
    NMI = normalized_mutual_info_score(target, y_predict)
    ARI = adjusted_rand_score(target, y_predict)
    print(" ACC = ", ACC)
    print(" NMI = ", NMI)
    print(" ARI = ", ARI)


if __name__ == '__main__':
    photo_name = []
    target = []
    total_photo = pd.DataFrame()
    getinfo()
    result, y_predict = kmeans()
    score()
    draw()





