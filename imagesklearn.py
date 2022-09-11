# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import matplotlib.pyplot
import numpy
import scipy
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy
import PIL.Image

def moonswork():
    matplotlib.pyplot.figure("yichi")
    matplotlib.pyplot.subplot(1,2,1)
    moons_X, moons_Y = make_moons(n_samples=400, noise=0.1)
    matplotlib.pyplot.title("make_circle")
    matplotlib.pyplot.scatter(moons_X[:, 0], moons_X[:, 1], s=100, marker="o", edgecolors='m', c=moons_Y, cmap='viridis')

    # 以下是第三个project中添加的内容
    matplotlib.pyplot.subplot(1,2,2)
    kmeans = KMeans(n_clusters=2)
    # 两个聚类中心 ↑
    kmeans.fit(moons_X)
    kmeans_y = kmeans.predict(moons_X)
    matplotlib.pyplot.title("make_circle by kmeans")
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
    ##三大指标
    acc = accuracy_score(circle_Y, kmeans_y)
    nmi = normalized_mutual_info_score(circle_Y, kmeans_y)
    ari = adjusted_rand_score(circle_Y, kmeans_y)

    print("ACC=", acc)
    print("NMI=", nmi)
    print("ARI=", ari)



    matplotlib.pyplot.show()

circlework()


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
                newpic.putpixel((i, j), (60, 0, 220))
            elif tmp[i][j] == 4:
                newpic.putpixel((i, j), (249, 219, 87))
            elif tmp[i][j] == 5:
                newpic.putpixel((i, j), (167, 255, 167))
            elif tmp[i][j] == 6:
                newpic.putpixel((i, j), (216, 109, 216))
    title = "k=" + str(t)
    plt.title(title)
    plt.imshow(newpic)
    plt.axis('off')
plt.show()




