import random
import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

left = 6;right = 7
with open(r'E:\lab_project\phos_data\CDK_merge.fasta') as in_f:
    with open(r'E:\lab_project\phos_data\human_phosphoplus.csv') as in_f2:
        #######################去除SS7蛋白质结构预测结果#################
        no_line3 = []
        for m, n in enumerate(in_f.readlines()):
            if (m - 3) % 6 != 0:
                no_line3.append(n.strip('\n'))


        h=0;g=0     ##########h表示激酶的磷酸化位点数目；g表示去重后的激酶底物数目
        sub=[];all_motif=[]
        positive1={};positive2={};positive3={}
        for i in in_f2.readlines():                         #############迭代激酶位点文件
            if 'CDK' in i.split('\t')[1]:                   ############找出某个激酶的所有条目
                h+=1                                        ###########用h进行计数
                for m,n in enumerate(no_line3):             ############迭代某个激酶底物的结构和序列信息
                    if '|' in n:                            ###########找出结构序列信息所在的行
                        if i.split('\t')[6]==n.split('|')[1]:   #############原代码判断条件错误，没有加上CDK激酶这一限制条件
                            g+=1                            ##########统计去重之后的底物位点数目
                            print(h,g)                      ###########观察去重前后的走势
                            if n.split('|')[1] not in sub:  #########统计某个激酶所有的底物蛋白
                                sub.append(n.split('|')[1])
                            postion = int(i.split('\t')[9][1:])         #############拿到底物的具体位点
                            if postion > left:              ###########位点不在最前端
                                if len(no_line3[m + 1]) - postion > 5:
                                    if len(no_line3[m + 1]) - postion == left:  #######找出motif刚好在蛋白质末端的数量
                                        #c += 1  # 找出motif刚好在蛋白质末端的数量
                                        positive1[no_line3[m + 1][postion - right:]] = no_line3[m + 2][postion - right:]
                                        positive2[no_line3[m + 1][postion - right:]] = no_line3[m + 3][postion - right:]
                                        positive3[no_line3[m + 1][postion - right:]] = no_line3[m + 4][postion - right:]
                                        # if len(n[int(i.split('\t')[9][1:]-left:])!=13:
                                        # print (n[int(i.split('\t')[9][1:]-left:])
                                        if no_line3[m + 1][postion - right:] not in all_motif:  # 添加无重复的motif
                                            all_motif.append(no_line3[m + 1][postion - right:])
                                    else:
                                        #d += 1  # 找出motif在蛋白质中间的数量
                                        positive1[no_line3[m + 1][postion - right:postion + left]] = no_line3[m + 2][
                                                                                                     postion - right:postion + left]
                                        positive2[no_line3[m + 1][postion - right:postion + left]] = no_line3[m + 3][
                                                                                                     postion - right:postion + left]
                                        positive3[no_line3[m + 1][postion - right:postion + left]] = no_line3[m + 4][
                                                                                                     postion - right:postion + left]
                                        # if len(n[int(i.split('\t')[9][1:]-left:int(i.split('\t')[9][1:]+6])!=13:
                                        # print (n[int(i.split('\t')[9][1:]-left:int(i.split('\t')[9][1:]+6])
                                        if no_line3[m + 1][
                                           postion - right:postion + left] not in all_motif:  # 添加无重复的motif
                                            all_motif.append(no_line3[m + 1][postion - right:postion + left])
                                else:
                                    #e += 1  # 找出在蛋白质末端且需要添加X的motif的数量
                                    positive1[no_line3[m + 1][postion - right:] + (
                                                postion + left - len(no_line3[m + 1])) * 'X'] = no_line3[m + 2][
                                                                                                postion - right:] + (
                                                                                                            postion + left - len(
                                                                                                        no_line3[
                                                                                                            m + 1])) * 'X'  # 为什么+9才能正常运行？
                                    positive2[no_line3[m + 1][postion - right:] + (
                                                postion + left - len(no_line3[m + 1])) * 'X'] = no_line3[m + 3][
                                                                                                postion - right:] + (
                                                                                                            postion + left - len(
                                                                                                        no_line3[
                                                                                                            m + 1])) * 'X'
                                    positive3[no_line3[m + 1][postion - right:] + (
                                                postion + left - len(no_line3[m + 1])) * 'X'] = no_line3[m + 4][
                                                                                                postion - right:] + (
                                                                                                            postion + left - len(
                                                                                                        no_line3[
                                                                                                            m + 1])) * 'X'
                                    if len(no_line3[m + 1][postion - right:] + (
                                            postion + left - len(no_line3[m + 1])) * 'X') != 13:
                                        print(no_line3[m + 1][postion - right:] + (
                                                    postion + left - len(no_line3[m + 1])) * 'X')  # 输出异常数据

                                    if no_line3[m + 1][postion - right:] + (postion + left - len(
                                            no_line3[m + 1])) * 'X' not in all_motif:  # 添加无重复的motif
                                        all_motif.append(no_line3[m + 1][postion - right:] + (
                                                    postion + left - len(no_line3[m + 1])) * 'X')
                            else:
                                #f += 1  # 找出在蛋白质前端（添加X）的motif的数量

                                positive1['X' * (right - postion) + no_line3[m + 1][:postion + left]] = 'X' * (
                                            right - postion) + no_line3[m + 2][:postion + left]
                                positive2['X' * (right - postion) + no_line3[m + 1][:postion + left]] = 'X' * (
                                            right - postion) + no_line3[m + 3][:postion + left]
                                positive3['X' * (right - postion) + no_line3[m + 1][:postion + left]] = 'X' * (
                                            right - postion) + no_line3[m + 4][:postion + left]
                                # if len('X'*(left-postion+n[:int(i.split('\t')[9][1:]+6])!=13:
                                # print ('X'*(left-postion+n[:int(i.split('\t')[9][1:]+6])
                                if 'X' * (right - postion) + no_line3[m + 1][:postion + left] not in all_motif:
                                    all_motif.append('X' * (right - postion) + no_line3[m + 1][:postion + left])
    in_f2.close()
in_f.close()
print('the number of kinase substrate:',len(sub))

all_dataset1={};all_dataset2={};all_dataset3={}
with open(r'E:\lab_project\phos_data\RegPhos.fasta')as in_f:
    with open(r'E:\lab_project\phos_data\CDK_merge.fasta') as in_f2:
        #######################去除SS7蛋白质结构预测结果#################
        lst = in_f.readlines()
        no_line3 = []
        for m, n in enumerate(in_f2.readlines()):
            if (m - 3) % 6 != 0:
                no_line3.append(n.strip('\n'))


        for i in sub:               ###############迭代某个激酶的所有底物
            sites=[]
            for x in lst:           ###############迭代包含有大量磷酸化位点的文件
                if i==x.split('\t')[1]:
                    sites.append(int(x.split('\t')[2]))
            # print(sites)
            for m, n in enumerate(no_line3):
                if '|' in n:
                    if i == n.split('|')[1]:
                        for pos,aa in enumerate(no_line3[m+1]):
                            if aa in ['Y','S','T'] and pos+1 not in sites:
                                if pos > 5:
                                    if len(no_line3[m + 1]) - pos > left:
                                        if len(no_line3[m + 1]) - pos == right:
                                            all_dataset1[no_line3[m + 1][pos - left:]] = no_line3[m + 2][pos - left:]
                                            all_dataset2[no_line3[m + 1][pos - left:]] = no_line3[m + 3][pos - left:]
                                            all_dataset3[no_line3[m + 1][pos - left:]] = no_line3[m + 4][pos - left:]

                                        else:
                                            all_dataset1[no_line3[m + 1][pos - left:pos + right]] = no_line3[m + 2][
                                                                                                    pos - left:pos + right]
                                            all_dataset2[no_line3[m + 1][pos - left:pos + right]] = no_line3[m + 3][
                                                                                                    pos - left:pos + right]
                                            all_dataset3[no_line3[m + 1][pos - left:pos + right]] = no_line3[m + 4][
                                                                                                    pos - left:pos + right]
                                    else:
                                        all_dataset1[
                                            no_line3[m + 1][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'] = \
                                        no_line3[m + 2][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'
                                        all_dataset2[
                                            no_line3[m + 1][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'] = \
                                        no_line3[m + 3][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'
                                        all_dataset3[
                                            no_line3[m + 1][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'] = \
                                        no_line3[m + 4][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'
                                else:
                                    all_dataset1['X' * (left - pos) + no_line3[m + 1][:pos + right]] = 'X' * (
                                                left - pos) + no_line3[m + 2][:pos + right]
                                    all_dataset2['X' * (left - pos) + no_line3[m + 1][:pos + right]] = 'X' * (
                                                left - pos) + no_line3[m + 3][:pos + right]
                                    all_dataset3['X' * (left - pos) + no_line3[m + 1][:pos + right]] = 'X' * (
                                                left - pos) + no_line3[m + 4][:pos + right]

    in_f2.close()
in_f.close()

print(len(all_dataset1))

final_vec1 = [];final_vec2 = []
T_positive = [acid for acid in positive1.keys() if acid[left] in ['T', 'Y', 'S']]
T_positive2 = [i for i in T_positive if 'X' not in i]

T_all = [acid for acid in all_dataset1.keys() if acid[left] in ['T', 'Y', 'S']]
T_all_negative = [i for i in T_all if i not in T_positive]
T_all_negative2 = [i for i in T_all_negative if 'X' not in i]
T_negative2 = random.sample(T_all_negative2, len(T_positive2))
all_dataset = T_negative2 + T_positive2
print(len(T_positive2))


########################H为helix的aa，E为beta的aa，C为coil的aa；B为深埋内部的aa，M为位于中部aa，E为暴露外界的aa；*为无序aa，.为有序aa#####################################
aa=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
structure=['C','H','E','X'];solvent=['X','E','M','B'];order=['X','*','.']
final_vec3=[];final_vec4=[]


for n in T_positive2:
    vec=[]
    for a in n:
        for x in aa:
            if a==x:
                if x!='X':
                    vec.append(1)
                else:
                    vec.append(0.5)
            else:
                vec.append(0)
    final_vec3.append(vec)

for n in T_negative2:
    vec = []
    for a in n:
        for x in aa:
            if a == x:
                if x != 'X':
                    vec.append(1)
                else:
                    vec.append(0.5)
            else:
                vec.append(0)
    final_vec4.append(vec)

vec2 = np.array(final_vec3+ final_vec4)
vec_label2 = np.array([1] * len(final_vec3) + [0] * len(final_vec4))

#########svm###########
clf=svm.LinearSVC()
scores = model_selection.cross_val_score(clf, vec2, y=vec_label2, cv=5)
print('Per accuracy in 5-fold CV:')
print(scores)
print("Accuracy of svm: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))