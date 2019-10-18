#coding=utf8

#Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt


print(tf.__path__)
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


#################导入数据集################
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_labels)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

###############预处理数据####################

##检查训练集中的第一张图像
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

##将这些像素值缩小到0到1之间，然后将其送到神经网络模型
train_images = train_images / 255.0
test_images = test_images / 255.0

##显示训练集中的前25张图象
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


################构建神经网络##########################

##设置层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

##编译模型
model.compile(loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

##训练模型
model.fit(train_images, train_labels, epochs=5)

##评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


################使用模型进行预测########################
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]

##将预测结果绘制成图，查看某一张图分别属于十种类别可能性大的大小
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

##查看第0张、第12张图像的预测结果
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
#
# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()

##Plot the first X test images, their predicted label, and the true label
##Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

