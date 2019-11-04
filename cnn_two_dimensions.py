#coding=utf8


##使用cnn处理二维数据
import os
import random
import numpy as np
import gensim.models as gm
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

os.chdir(r'E:\lab_project')

with open(r'E:\lab_project\phos_data\CDK_merge.fasta') as in_f:
    with open(r'E:\lab_project\phos_data\human_phosphoplus.csv') as in_f2:
#######################去除SSright蛋白质结构预测结果#################
        no_line3=[]
        for m,n in enumerate(in_f.readlines()):
            if (m-3)%6!=0:
                no_line3.append(n.strip('\n'))
        #print (len(no_line3))


#################检查是否有蛋白质长度小于13##############
        b=0
        for i in no_line3:
            if '>' not in i and len(i)<=13:
                b+=1
        #print (b)

#######################将各种预测信息分到不同字典中################
        all_motif=[];c=0;d=0;e=0;f=0;g=0;h=0
        positive1={};positive2={};positive3={}
        all_dataset1={};all_dataset2={};all_dataset3={}
        left=6;right=7
        for i in in_f2.readlines():
            if 'CDK' in i.split('\t')[1]:
                h+=1
            for m,n in enumerate(no_line3):
                if '|' in n:
                    if 'CDK' in i.split('\t')[1] and i.split('\t')[6]==n.split('|')[1]:				#############原代码判断条件错误，没有加上CDK激酶这一限制条件
                        g+=1
                        for pos,aa in enumerate(no_line3[m+1]):
                            if aa in ['Y','S','T']:
                                if pos>5:
                                    if len(no_line3[m+1])-pos>left:
                                        if len(no_line3[m+1])-pos==right:
                                            all_dataset1[no_line3[m+1][pos-left:]]=no_line3[m+2][pos-left:]
                                            all_dataset2[no_line3[m+1][pos-left:]]=no_line3[m+3][pos-left:]
                                            all_dataset3[no_line3[m+1][pos-left:]]=no_line3[m+4][pos-left:]

                                        else:
                                            all_dataset1[no_line3[m+1][pos-left:pos+right]]=no_line3[m+2][pos-left:pos+right]
                                            all_dataset2[no_line3[m+1][pos-left:pos+right]]=no_line3[m+3][pos-left:pos+right]
                                            all_dataset3[no_line3[m+1][pos-left:pos+right]]=no_line3[m+4][pos-left:pos+right]
                                    else:
                                        all_dataset1[no_line3[m+1][pos-left:]+(pos+right-len(no_line3[m+1]))*'X']=no_line3[m+2][pos-left:]+(pos+right-len(no_line3[m+1]))*'X'
                                        all_dataset2[no_line3[m+1][pos-left:]+(pos+right-len(no_line3[m+1]))*'X']=no_line3[m+3][pos-left:]+(pos+right-len(no_line3[m+1]))*'X'
                                        all_dataset3[no_line3[m+1][pos-left:]+(pos+right-len(no_line3[m+1]))*'X']=no_line3[m+4][pos-left:]+(pos+right-len(no_line3[m+1]))*'X'
                                else:
                                    all_dataset1['X'*(left-pos)+no_line3[m+1][:pos+right]]='X'*(left-pos)+no_line3[m+2][:pos+right]
                                    all_dataset2['X'*(left-pos)+no_line3[m+1][:pos+right]]='X'*(left-pos)+no_line3[m+3][:pos+right]
                                    all_dataset3['X'*(left-pos)+no_line3[m+1][:pos+right]]='X'*(left-pos)+no_line3[m+4][:pos+right]



                        postion=int(i.split('\t')[9][1:])
                        if postion>left:
                            if len(no_line3[m+1])-postion>5:
                                if len(no_line3[m+1])-postion==left:
                                    c+=1																						#找出motif刚好在蛋白质末端的数量
                                    positive1[no_line3[m+1][postion-right:]]=no_line3[m+2][postion-right:]
                                    positive2[no_line3[m+1][postion-right:]]=no_line3[m+3][postion-right:]
                                    positive3[no_line3[m+1][postion-right:]]=no_line3[m+4][postion-right:]
                                    # if len(n[int(i.split('\t')[9][1:]-left:])!=13:
                                        # print (n[int(i.split('\t')[9][1:]-left:])
                                    if no_line3[m+1][postion-right:] not in all_motif:														#添加无重复的motif
                                        all_motif.append(no_line3[m+1][postion-right:])
                                else:
                                    d+=1																							#找出motif在蛋白质中间的数量
                                    positive1[no_line3[m+1][postion-right:postion+left]]=no_line3[m+2][postion-right:postion+left]
                                    positive2[no_line3[m+1][postion-right:postion+left]]=no_line3[m+3][postion-right:postion+left]
                                    positive3[no_line3[m+1][postion-right:postion+left]]=no_line3[m+4][postion-right:postion+left]
                                    # if len(n[int(i.split('\t')[9][1:]-left:int(i.split('\t')[9][1:]+6])!=13:
                                        # print (n[int(i.split('\t')[9][1:]-left:int(i.split('\t')[9][1:]+6])
                                    if no_line3[m+1][postion-right:postion+left] not in all_motif:											#添加无重复的motif
                                        all_motif.append(no_line3[m+1][postion-right:postion+left])
                            else:
                                e+=1																								#找出在蛋白质末端且需要添加X的motif的数量
                                positive1[no_line3[m+1][postion-right:]+(postion+left-len(no_line3[m+1]))*'X']=no_line3[m+2][postion-right:]+(postion+left-len(no_line3[m+1]))*'X'			#为什么+9才能正常运行？
                                positive2[no_line3[m+1][postion-right:]+(postion+left-len(no_line3[m+1]))*'X']=no_line3[m+3][postion-right:]+(postion+left-len(no_line3[m+1]))*'X'
                                positive3[no_line3[m+1][postion-right:]+(postion+left-len(no_line3[m+1]))*'X']=no_line3[m+4][postion-right:]+(postion+left-len(no_line3[m+1]))*'X'
                                if len(no_line3[m+1][postion-right:]+(postion+left-len(no_line3[m+1]))*'X')!=13:
                                    print (no_line3[m+1][postion-right:]+(postion+left-len(no_line3[m+1]))*'X')											#输出异常数据

                                if no_line3[m+1][postion-right:]+(postion+left-len(no_line3[m+1]))*'X' not in all_motif:									#添加无重复的motif
                                    all_motif.append(no_line3[m+1][postion-right:]+(postion+left-len(no_line3[m+1]))*'X')
                        else:
                            f+=1																									#找出在蛋白质前端（添加X）的motif的数量

                            positive1['X'*(right-postion)+no_line3[m+1][:postion+left]]='X'*(right-postion)+no_line3[m+2][:postion+left]
                            positive2['X'*(right-postion)+no_line3[m+1][:postion+left]]='X'*(right-postion)+no_line3[m+3][:postion+left]
                            positive3['X'*(right-postion)+no_line3[m+1][:postion+left]]='X'*(right-postion)+no_line3[m+4][:postion+left]
                            # if len('X'*(left-postion+n[:int(i.split('\t')[9][1:]+6])!=13:
                                # print ('X'*(left-postion+n[:int(i.split('\t')[9][1:]+6])
                            if 'X'*(right-postion)+no_line3[m+1][:postion+left] not in all_motif:
                                all_motif.append('X'*(right-postion)+no_line3[m+1][:postion+left])

        print (g,h)	########源文件中某个激酶对应的底物的所有条目
#################鉴定是否存在异常数据########################
        error=0
        for i in positive1.keys():
            if len(i)!=13:
                error+=1
                #print (i,all_dataset1[i])
                del positive1[i]
        for i in positive2.keys():
            if len(i)!=13:
                del positive2[i]
        for i in positive3.keys():
            if len(i)!=13:
                del positive3[i]
        print ('the number of abnormal data:',error)
        print ('all identified S/Y/T sites:',len(positive1.keys()));print ('all (un)identified S/Y/T sites:',len(all_dataset3.values()))	#结果有1left303个motif，但在原始文件中存在1leftleft33个位点，因此应该有430个重复位点
        print (len(all_motif))			##############所有无重复的磷酸化motif
    in_f2.close()
in_f.close()



final_vec1=[]
T_positive=[acid for acid in positive1.keys() if acid[left] in ['T','Y','S']]
T_positive2=[i for i in T_positive if 'X' not in i]
final_vec2=[]
T_all=[acid for acid in all_dataset1.keys() if acid[left] in ['T','Y','S']]
T_all_negative=[i for i in T_all if i not in T_positive]
T_all_negative2=[i for i in T_all_negative if 'X' not in i]
T_negative2=random.sample(T_all_negative2,len(T_positive2))
all_dataset=T_negative2+T_positive2
print (len(T_positive2))

# ############在CDK激酶的底物中，存在一个不是SYT的磷酸化位点：CMGGMNRrPILTIIT；另外，P6right431底物的S11位点被磷酸化，但原文件统计时，写成了s10
# T_abnormal=[i for i in positive1.keys() if i[left] not in ['T','S','Y']]
# print (T_abnormal)
#
# #######################H为helix的aa，E为beta的aa，C为coil的aa；B为深埋内部的aa，M为位于中部aa，E为暴露外界的aa；*为无序aa，.为有序aa#####################################
# aa=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
# structure=['C','H','E','X'];solvent=['X','E','M','B'];order=['X','*','.']
# final_vec3=[];final_vec4=[]
#
#
# for n in T_positive2:
#     vec=[]
#     for a in n:
#         for x in aa:
#             if a==x:
#                 if x!='X':
#                     vec.append(1)
#                 else:
#                     vec.append(0.5)
#             else:
#                 vec.append(0)
#     vec.append(1)
#     vec.append(0)
#     final_vec3.append(vec)
#
# print (len(final_vec3[1]))
#
#
# for n in T_negative2:
#     vec=[]
#     for a in n:
#         for x in aa:
#             if a==x:
#                 if x!='X':
#                     vec.append(1)
#                 else:
#                     vec.append(0.5)
#             else:
#                 vec.append(0)
#     vec.append(0.)
#     vec.append(1)
#     final_vec4.append(vec)
# print (final_vec4[1])


mod = gm.Doc2Vec.load(r'E:\lab_project\phos\model\doc2vector.bin')

##使用模型处理激酶底物peptide数据
docs=[]
for i in T_positive2:
    doc=[]
    for x in range(10):
       doc.append(i[x:x+3])
    doc.append(i[-3:])
    docs.append(doc)
print(docs)

for i in docs:
    final_vec1.append(list(mod.infer_vector(i)))
# print(mod.infer_vector(['SPL', 'PLK', 'LKA', 'KAY', 'AYT', 'YTP', 'TPV', 'PVV', 'VVV', 'VVT', 'VTL']))
final_vec3=[i.append(1) for i in final_vec1]
final_vec4=[i.append(0) for i in final_vec1]
print(len(final_vec1[1]))


docs=[]
for i in T_negative2:
    doc=[]
    for x in range(10):
       doc.append(i[x:x+3])
    doc.append(i[-3:])
    docs.append(doc)
print(docs)

for i in docs:
    final_vec2.append(list(mod.infer_vector(i)))
print(type(final_vec1))
final_vec4=[i.append(0) for i in final_vec2]
final_vec3=[i.append(1) for i in final_vec2]


vec1 = np.array(final_vec1+ final_vec2)
#vec_label1 = np.array([1] * len(final_vec3) + [0] * len(final_vec4))
print(vec1.shape)
np.random.shuffle(vec1)
print(vec1[:10])

# VALIDATION_SIZE=144
test_size=252
# validation_images = vec1[:, :-2][:VALIDATION_SIZE]
# validation_labels = vec1[:, -2:][:VALIDATION_SIZE]
train = vec1[:, :][test_size:]
train_images = train[:, :-2]
train_labels = train[:, -2:]
# test_images = vec1[:, :-2][VALIDATION_SIZE:test_size]
# test_labels = vec1[:, -2:][VALIDATION_SIZE:test_size]

test_images=vec1[:,:-2][:test_size]
test_labels=vec1[:,-2:][:test_size]


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32,[None, 100],name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 2],name='y_input')

##reshape image数据
#To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height,
#and the final dimension corresponding to the number of color channels.-1表示任意数量的样本数
    x_image = tf.reshape(x, [-1,10,10,1])

##定义weights和bias
#----Weight Initialization---#
#One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
    
##定义卷积层和最大池化层
#Convolution and Pooling
#Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input.
#Our pooling is plain old max pooling over 2x2 blocks
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

##搭建第一个卷积层，使用32个5*5的filtering，然后maxpooling
#----first convolution layer----#
#he convolution will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32].
#The first two dimensions are the patch size,
#the next is the number of input channels, and the last is the number of output channels.
with tf.name_scope('layer1'):
    with tf.name_scope('W1'):
        W_conv1 = weight_variable([3,3,1,32])
        tf.summary.histogram('layer1_W1',W_conv1)
    #We will also have a bias vector with a component for each output channel.
    with tf.name_scope('b1'):
        b_conv1 = bias_variable([32])
        tf.summary.histogram('layer1_b1',b_conv1)

#We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
#The max_pool_2x2 method will reduce the image size to 14x14.
    with tf.name_scope('conv1'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)


##第二个卷积层，使用64个filter，然后最大池化
#----second convolution layer----#
#The second layer will have 64 features for each 5x5 patch and input size 32.
with tf.name_scope('layer2'):
    with tf.name_scope('W2'):
        W_conv2 = weight_variable([3,3,32,64])
    with tf.name_scope('b2'):
        b_conv2 = bias_variable([64])

    with tf.name_scope('conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)


##构建全链接层
#----fully connected layer----#
#Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image
with tf.name_scope('full_layer'):
    W_fc1 = weight_variable([3*3*64, 50])
    b_fc1 = bias_variable([50])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)


##添加dropout
#-----dropout------#
#To reduce overfitting, we will apply dropout before the readout layer.
#We create a placeholder for the probability that a neuron's output is kept during dropout.
#This allows us to turn dropout on during training, and turn it off during testing.
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)


##输出层
#----read out layer----#
with tf.name_scope('outlayer'):
    W_fc2 = weight_variable([50,2])
    b_fc2 = bias_variable([2])
    y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2


##训练和评估
#------train and evaluate----#
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar('cross_entroy',cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)), tf.float32))
    tf.summary.scalar('accuracy',accuracy)

with tf.Session() as sess:
    writer=tf.summary.FileWriter('logs/',sess.graph)
    merged=tf.summary.merge_all()
    tf.global_variables_initializer().run()
    index_in_epoch = 0;end = index_in_epoch
    for i in range(1000):
        if end < (len(vec1) -252):
            # for m in range(int(total_batch)):
            start = index_in_epoch
            index_in_epoch += 50        #batch_size
            end = index_in_epoch
            batch_xs, batch_ys = train_images[start:end], train_labels[start:end]  # max(x) = 1, min(x) = 0
            if i % 100 ==0:
                train_accuracy = accuracy.eval(feed_dict = {x: batch_xs,y_: batch_ys,keep_prob: 1.})
                acc = sess.run(merged,feed_dict = {x: batch_xs,y_: batch_ys,keep_prob: 1.})
                writer.add_summary(acc,i)
                print('setp {},the train accuracy: {}'.format(i, train_accuracy))
            _ ,loss = sess.run([train_step,merged],feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            writer.add_summary(loss,i)
        else:
            index_in_epoch=0;end=index_in_epoch
            np.random.shuffle(train)
    test_accuracy = accuracy.eval(feed_dict = {x: test_images, y_: test_labels, keep_prob: 1.})
    print('the test accuracy :{}'.format(test_accuracy))
    # saver = tf.train.Saver()
    # path = saver.save(sess, './my_net/mnist_deep.ckpt')
    # print('save path: {}'.format(path))
