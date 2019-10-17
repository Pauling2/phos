#coding=utf8

#Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os

#相关库
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
os.chdir(r'C:\Users\xzw\.keras\datasets\fashion-mnist')
# print(os.getcwd())
# cache1=os.path.join(os.path.expanduser('~'), '.keras')
# datadir_base=os.path.expanduser(cache1)
# print(os.access(datadir_base,os.W_OK))
# cache_subdir=os.path.join('datasets', 'fashion-mnist')
# datadir = os.path.join(datadir_base, cache_subdir)
# print(datadir)
# print(os.path.join(datadir, 'train-labels-idx1-ubyte.gz'))
# A=np.load('mnist.npz')

##导入数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_labels)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

##预处理数据
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()







