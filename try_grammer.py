#coding=utf8


##############各种语法，内容的尝试

####动态传参：(单个或多个）列表传入时，会被套在元组中传入处理

# lst=[[1,2,3,4],[2,3,4,5],[3,4,5,6]]
# def func(*args):
#     print args
#     sum=0
#     for i in args:
#         for x in i:
#             sum+=x
#     print sum
#
# if __name__=='__main__':
#     func([1,2,3,4],[2,3,4,5],[3,4,5,6])
#     lst=([11,2,3],2,3)
#     print lst[0][1]


#input_num = int(input())


# def func(num):
#     x = 0;
#     item = 0
#     while x < num:
#         y = 0
#         while y < num:
#             if x ** 2 + y ** 2 == num:
#                 if x!=0 and y!=0:
#                     item += 4
#                 else:
#                     item+=2
#             y += 1
#         x += 1
#     print item
#
#
# if __name__ == '__main__':
#     func(input_num)


# while 1:
#     i = raw_input()
#     if i == "END":
#         break
#     exec("print("+i+")")

# import numpy as np
# si=input()
# # print isinstance(si,str)
# # print si.split(' ')
# data=[]
# def func():
#     i=0
#     while i<si-1:
#         you_data=raw_input()
#         once=[int(x) for x in you_data.split(' ')]
#         data.append(once)
#         i+=1
#     np1=np.array(data)
#     #print type(list(np1[:,1]))
#     for i in list(np1[:,0]):
#         for x in list(np1[:, 1]):
#
# func()


######判断布尔值
# lst=[]
# print bool(lst)


##统计某个激酶家族的所有激酶
import re

# def func(filename2):
#     with open(filename2)as in_f:
#         category = in_f.readlines();
#         kinases=['PKA','PKC','ATM','CDK','CK2','Src'];
#         for x in kinases:
#             group = []
#             for i in category:
#                 if i.split(' ')[2]==x and i.split(' ')[0] not in group:
#                     group.append(i.split(' ')[0])
#             print (group)
#
#     in_f.close()
#
# if __name__=='__main__':
#     func(r'E:\python_project\data\cate_kinase.csv')



##查看两种提取激酶的底物方法所提取底物的不同
# with open(r'C:\Users\xzw\Desktop\human_phosphoplus.csv') as in_f:
#     h=0;g=0;h1=[];h2=[];diff=[]
#     for i in in_f.readlines():
#         if i.split('\t')[1] in ['PKACA', 'PKACB', 'PKACG', 'PRKX', 'PRKY']:
#             h += 1
#             h1.append(i.split('\t')[1])
#         # if 'CDK' in i.split('\t')[1]:
#         #     g+=1
#         #     h2.append(i.split('\t')[1])
#     # for i in h2:
#     #     if i not in h1:
#     #         diff.append(i)
#     print (h);print (h1)



##尝试doc模型
import gensim.models as gm
import random
import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt

with open(r'C:\Users\xzw\Desktop\CDK_merge.fasta') as in_f:
    with open(r'C:\Users\xzw\Desktop\human_phosphoplus.csv') as in_f2:
        #######################去除SS8蛋白质结构预测结果#################
        no_line3 = []
        for m, n in enumerate(in_f.readlines()):
            if (m - 3) % 6 != 0:
                no_line3.append(n.strip('\n'))
        # print len(no_line3)
        
        #################检查是否有蛋白质长度小于13##############
        b = 0
        for i in no_line3:
            if '>' not in i and len(i) <= 13:
                b += 1
        # print 0
        
        #######################将各种预测信息分到不同字典中################
        all_motif = [];
        c = 0;
        d = 0;
        e = 0;
        f = 0;
        g = 0;
        h = 0
        positive1 = {};
        positive2 = {};
        positive3 = {}
        all_dataset1 = {};
        all_dataset2 = {};
        all_dataset3 = {}
        for i in in_f2.readlines():
            if 'CDK' in i.split('\t')[1]:
                h += 1
                for m, n in enumerate(no_line3):
                    if '|' in n:
                        if i.split('\t')[6] == n.split('|')[1]:  #############原代码判断条件错误，没有加上CDK激酶这一限制条件;现在把这一条件判断的缩进往后一步，以免再写一遍CDK激酶这一限定条件
                            g += 1
                            for pos, aa in enumerate(no_line3[m + 1]):
                                if aa in ['Y', 'S', 'T']:
                                    if pos > 5:
                                        if len(no_line3[m + 1]) - pos > 6:
                                            if len(no_line3[m + 1]) - pos == 7:
                                                all_dataset1[no_line3[m + 1][pos - 6:]] = no_line3[m + 2][pos - 6:]
                                                all_dataset2[no_line3[m + 1][pos - 6:]] = no_line3[m + 3][pos - 6:]
                                                all_dataset3[no_line3[m + 1][pos - 6:]] = no_line3[m + 4][pos - 6:]
                                            
                                            else:
                                                all_dataset1[no_line3[m + 1][pos - 6:pos + 7]] = no_line3[m + 2][
                                                                                                 pos - 6:pos + 7]
                                                all_dataset2[no_line3[m + 1][pos - 6:pos + 7]] = no_line3[m + 3][
                                                                                                 pos - 6:pos + 7]
                                                all_dataset3[no_line3[m + 1][pos - 6:pos + 7]] = no_line3[m + 4][
                                                                                                 pos - 6:pos + 7]
                                        else:
                                            all_dataset1[
                                                no_line3[m + 1][pos - 6:] + (pos + 7 - len(no_line3[m + 1])) * 'X'] = \
                                            no_line3[m + 2][pos - 6:] + (pos + 7 - len(no_line3[m + 1])) * 'X'
                                            all_dataset2[
                                                no_line3[m + 1][pos - 6:] + (pos + 7 - len(no_line3[m + 1])) * 'X'] = \
                                            no_line3[m + 3][pos - 6:] + (pos + 7 - len(no_line3[m + 1])) * 'X'
                                            all_dataset3[
                                                no_line3[m + 1][pos - 6:] + (pos + 7 - len(no_line3[m + 1])) * 'X'] = \
                                            no_line3[m + 4][pos - 6:] + (pos + 7 - len(no_line3[m + 1])) * 'X'
                                    else:
                                        all_dataset1['X' * (6 - pos) + no_line3[m + 1][:pos + 7]] = 'X' * (6 - pos) + \
                                                                                                    no_line3[m + 2][
                                                                                                    :pos + 7]
                                        all_dataset2['X' * (6 - pos) + no_line3[m + 1][:pos + 7]] = 'X' * (6 - pos) + \
                                                                                                    no_line3[m + 3][
                                                                                                    :pos + 7]
                                        all_dataset3['X' * (6 - pos) + no_line3[m + 1][:pos + 7]] = 'X' * (6 - pos) + \
                                                                                                    no_line3[m + 4][
                                                                                                    :pos + 7]
                            
                            postion = int(i.split('\t')[9][1:])
                            if postion > 6:
                                if len(no_line3[m + 1]) - postion > 5:
                                    if len(no_line3[m + 1]) - postion == 6:
                                        c += 1  # 找出motif刚好在蛋白质末端的数量
                                        positive1[no_line3[m + 1][postion - 7:]] = no_line3[m + 2][postion - 7:]
                                        positive2[no_line3[m + 1][postion - 7:]] = no_line3[m + 3][postion - 7:]
                                        positive3[no_line3[m + 1][postion - 7:]] = no_line3[m + 4][postion - 7:]
                                        # if len(n[int(i.split('\t')[9][1:]-7:])!=13:
                                        # print n[int(i.split('\t')[9][1:]-7:]
                                        if no_line3[m + 1][postion - 7:] not in all_motif:  # 添加无重复的motif
                                            all_motif.append(no_line3[m + 1][postion - 7:])
                                    else:
                                        d += 1  # 找出motif在蛋白质中间的数量
                                        positive1[no_line3[m + 1][postion - 7:postion + 6]] = no_line3[m + 2][
                                                                                              postion - 7:postion + 6]
                                        positive2[no_line3[m + 1][postion - 7:postion + 6]] = no_line3[m + 3][
                                                                                              postion - 7:postion + 6]
                                        positive3[no_line3[m + 1][postion - 7:postion + 6]] = no_line3[m + 4][
                                                                                              postion - 7:postion + 6]
                                        # if len(n[int(i.split('\t')[9][1:]-7:int(i.split('\t')[9][1:]+6])!=13:
                                        # print n[int(i.split('\t')[9][1:]-7:int(i.split('\t')[9][1:]+6]
                                        if no_line3[m + 1][postion - 7:postion + 6] not in all_motif:  # 添加无重复的motif
                                            all_motif.append(no_line3[m + 1][postion - 7:postion + 6])
                                else:
                                    e += 1  # 找出在蛋白质末端且需要添加X的motif的数量
                                    positive1[
                                        no_line3[m + 1][postion - 7:] + (postion + 6 - len(no_line3[m + 1])) * 'X'] = \
                                    no_line3[m + 2][postion - 7:] + (
                                                postion + 6 - len(no_line3[m + 1])) * 'X'  # 为什么+9才能正常运行？
                                    positive2[
                                        no_line3[m + 1][postion - 7:] + (postion + 6 - len(no_line3[m + 1])) * 'X'] = \
                                    no_line3[m + 3][postion - 7:] + (postion + 6 - len(no_line3[m + 1])) * 'X'
                                    positive3[
                                        no_line3[m + 1][postion - 7:] + (postion + 6 - len(no_line3[m + 1])) * 'X'] = \
                                    no_line3[m + 4][postion - 7:] + (postion + 6 - len(no_line3[m + 1])) * 'X'
                                    if len(no_line3[m + 1][postion - 7:] + (
                                            postion + 6 - len(no_line3[m + 1])) * 'X') != 13:
                                        print no_line3[m + 1][postion - 7:] + (
                                                    postion + 6 - len(no_line3[m + 1])) * 'X'  # 输出异常数据
                                    
                                    if no_line3[m + 1][postion - 7:] + (
                                            postion + 6 - len(no_line3[m + 1])) * 'X' not in all_motif:  # 添加无重复的motif
                                        all_motif.append(
                                            no_line3[m + 1][postion - 7:] + (postion + 6 - len(no_line3[m + 1])) * 'X')
                            else:
                                f += 1  # 找出在蛋白质前端（添加X）的motif的数量
                                
                                positive1['X' * (7 - postion) + no_line3[m + 1][:postion + 6]] = 'X' * (7 - postion) + \
                                                                                                 no_line3[m + 2][
                                                                                                 :postion + 6]
                                positive2['X' * (7 - postion) + no_line3[m + 1][:postion + 6]] = 'X' * (7 - postion) + \
                                                                                                 no_line3[m + 3][
                                                                                                 :postion + 6]
                                positive3['X' * (7 - postion) + no_line3[m + 1][:postion + 6]] = 'X' * (7 - postion) + \
                                                                                                 no_line3[m + 4][
                                                                                                 :postion + 6]
                                # if len('X'*(7-postion+n[:int(i.split('\t')[9][1:]+6])!=13:
                                # print 'X'*(7-postion+n[:int(i.split('\t')[9][1:]+6]
                                if 'X' * (7 - postion) + no_line3[m + 1][:postion + 6] not in all_motif:
                                    all_motif.append('X' * (7 - postion) + no_line3[m + 1][:postion + 6])
        
        print g, h  ########源文件中某个激酶对应的底物的所有条目
        #################鉴定是否存在异常数据########################
        error = 0
        for i in positive1.keys():
            if len(i) != 13:
                error += 1
                # print i,all_dataset1[i]
                del positive1[i]
        for i in positive2.keys():
            if len(i) != 13:
                del positive2[i]
        for i in positive3.keys():
            if len(i) != 13:
                del positive3[i]
        print 'the number of abnormal data:', error
        print 'all identified S/Y/T sites:', len(positive1.keys());
        print 'all (un)identified S/Y/T sites:', len(
            all_dataset3.values())  # 结果有17303个motif，但在原始文件中存在17733个位点，因此应该有430个重复位点
        print len(all_motif)  ##############所有无重复的磷酸化motif
        
        final_vec1 = []
        T_positive = [acid for acid in positive1.keys() if acid[6] in ['T', 'Y', 'S']]
        T_positive2 = [i for i in T_positive if 'X' not in i]
        final_vec2 = []
        T_all = [acid for acid in all_dataset1.keys() if acid[6] in ['T', 'Y', 'S']]
        T_all_negative = [i for i in T_all if i not in T_positive]
        T_all_negative2 = [i for i in T_all_negative if 'X' not in i]
        T_negative2 = random.sample(T_all_negative2, len(T_positive2))
        all_dataset = T_negative2 + T_positive2
        print len(T_positive2)
    
    in_f2.close()
in_f.close()

mod = gm.Doc2Vec.load(r'E:\python_project\model\doc2vector.bin')

docs=[]
for i in T_positive2:
    doc=[]
    for x in range(10):
       doc.append(i[x:x+3])
    doc.append(i[-3:])
    docs.append(doc)
print docs

for i in docs:
    final_vec1.append(list(mod.infer_vector(i)))
    
# print final_vec1

docs=[]
for i in T_negative2:
    doc=[]
    for x in range(10):
       doc.append(i[x:x+3])
    doc.append(i[-3:])
    docs.append(doc)
print docs

for i in docs:
    final_vec2.append(list(mod.infer_vector(i)))

vec2 = np.array(final_vec1 + final_vec2)
vec_label2 = np.array([1] * len(final_vec1) + [0] * len(final_vec2))

clf = svm.LinearSVC()
scores = model_selection.cross_val_score(clf, vec2, y=vec_label2, cv=5)
print('Per accuracy in 5-fold CV:')
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

cv = StratifiedKFold(n_splits=10)
classifier = svm.SVC(kernel='linear', probability=True)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
cnt=0
for i, (train, test) in enumerate(cv.split(vec2,vec_label2)):
    cnt=cnt+1
    probas_ = classifier.fit(vec2[train], vec_label2[train]).predict_proba(vec2[test])
    fpr, tpr, thresholds = roc_curve(vec_label2[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0


# Plot ROC curve.
mean_tpr /= cnt
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, '-', label='Mean ROC %s(area = %0.2f)' % ('doc_CDK',mean_auc), lw=2)

plt.xlim([0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('ROC of CDK_kinase')
plt.show()









