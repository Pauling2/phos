#coding=utf-8

# with open(r'C:\Users\xzw\Desktop\human_phosphoplus.csv') as in_f:
	# noredun=[]
	# for i in in_f.readlines():
		# if i .split('\t')[1]=='ATM' and i.split('\t')[6] not in noredun:
			# noredun.append(i.split('\t')[6])
	# print (noredun)
	
	
# with open(r'C:\Users\xzw\Desktop\ATM.txt','w') as out_f:
	# for i in noredun:
		# out_f.write(i+'\n')
		
		
		
import random
import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

		
with open(r'E:\lab_project\phos_data\CK2_merge.fasta') as in_f:
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
			if 'CK2' in i.split('\t')[1]:
				h+=1
			for m,n in enumerate(no_line3):
				if '|' in n:
					if 'CK2' in i.split('\t')[1] and i.split('\t')[6]==n.split('|')[1]:				#############原代码判断条件错误，没有加上CDK激酶这一限制条件
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
		


		
		final_vec1=[]
		T_positive=[acid for acid in positive1.keys() if acid[left] in ['T','Y','S']]
		T_positive2=[i for i in T_positive if 'X' not in i]
		final_vec2=[]
		T_all=[acid for acid in all_dataset1.keys() if acid[left] in ['T','Y','S']]
		T_all_negative=[i for i in T_all if i not in T_positive]
		T_all_negative2=[i for i in T_all_negative if 'X' not in i]
		T_negative2=random.sample(T_all_negative2,len(T_positive))
		all_dataset=T_negative2+T_positive2
		print (len(T_positive2))
		
############在CDK激酶的底物中，存在一个不是SYT的磷酸化位点：CMGGMNRrPILTIIT；另外，P6right431底物的S11位点被磷酸化，但原文件统计时，写成了s10	
		T_abnormal=[i for i in positive1.keys() if i[left] not in ['T','S','Y']]
		print (T_abnormal)


###############创建20个分字典及一个总字典######################		
		C={'C':9,'S':-1,'T':-1,'P':-3,'A':0,'G':-3,'N':-3,'D':-3,'E':-4,'Q':-3,'H':-3,'R':-3,'K':-3,'M':-1,'I':-1,'L':-1,'V':-1,'F':-2,'Y':-2,'W':-2}
		S={'C':-1,'S':4,'T':1,'P':-1,'A':1,'G':0,'N':1,'D':0,'E':0,'Q':0,'H':-1,'R':-1,'K':0,'M':-1,'I':-2,'L':-2,'V':-2,'F':-2,'Y':-2,'W':-3}
		T={'C':-1,'S':1,'T':4,'P':1,'A':-1,'G':1,'N':0,'D':1,'E':0,'Q':0,'H':0,'R':-1,'K':0,'M':-1,'I':-2,'L':-2,'V':-2,'F':-2,'Y':-2,'W':-3}
		P={'C':-3,'S':-1,'T':1,'P':left,'A':-1,'G':-2,'N':-1,'D':-1,'E':-1,'Q':-1,'H':-2,'R':-2,'K':-1,'M':-2,'I':-3,'L':-3,'V':-2,'F':-4,'Y':-3,'W':-4}
		A={'C':0,'S':1,'T':-1,'P':-1,'A':4,'G':0,'N':-1,'D':-2,'E':-1,'Q':-1,'H':-2,'R':-1,'K':-1,'M':-1,'I':-1,'L':-1,'V':-2,'F':-2,'Y':-2,'W':-3}
		G={'C':-3,'S':0,'T':1,'P':-2,'A':0,'G':6,'N':-2,'D':-1,'E':-2,'Q':-2,'H':-2,'R':-2,'K':-2,'M':-3,'I':-4,'L':-4,'V':0,'F':-3,'Y':-3,'W':-2}
		N={'C':-3,'S':1,'T':0,'P':-2,'A':-2,'G':0,'N':6,'D':1,'E':0,'Q':0,'H':-1,'R':0,'K':0,'M':-2,'I':-3,'L':-3,'V':-3,'F':-3,'Y':-2,'W':-4}
		D={'C':-3,'S':0,'T':1,'P':-1,'A':-2,'G':-1,'N':1,'D':6,'E':2,'Q':0,'H':-1,'R':-2,'K':-1,'M':-3,'I':-3,'L':-4,'V':-3,'F':-3,'Y':-3,'W':-4}
		E={'C':-4,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':2,'E':5,'Q':2,'H':0,'R':0,'K':1,'M':-2,'I':-3,'L':-3,'V':-3,'F':-3,'Y':-2,'W':-3}
		Q={'C':-3,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':0,'E':2,'Q':5,'H':0,'R':1,'K':1,'M':0,'I':-3,'L':-2,'V':-2,'F':-3,'Y':-1,'W':-2}
		H={'C':-3,'S':-1,'T':0,'P':-2,'A':-2,'G':-2,'N':1,'D':1,'E':0,'Q':0,'H':right,'R':0,'K':-1,'M':-2,'I':-3,'L':-3,'V':-2,'F':-1,'Y':2,'W':-2}
		R={'C':-3,'S':-1,'T':-1,'P':-2,'A':-1,'G':-2,'N':0,'D':-2,'E':0,'Q':1,'H':0,'R':5,'K':2,'M':-1,'I':-3,'L':-2,'V':-3,'F':-3,'Y':-2,'W':-3}
		K={'C':-3,'S':0,'T':0,'P':-1,'A':-1,'G':-2,'N':0,'D':-1,'E':1,'Q':1,'H':-1,'R':2,'K':5,'M':-1,'I':-3,'L':-2,'V':-3,'F':-3,'Y':-2,'W':-3}
		M={'C':-1,'S':-1,'T':-1,'P':-2,'A':-1,'G':-3,'N':-2,'D':-3,'E':-2,'Q':0,'H':-2,'R':-1,'K':-1,'M':5,'I':1,'L':2,'V':-2,'F':0,'Y':-1,'W':-1}
		I={'C':-1,'S':-2,'T':-2,'P':-3,'A':-1,'G':-4,'N':-3,'D':-3,'E':-3,'Q':-3,'H':-3,'R':-3,'K':-3,'M':1,'I':4,'L':2,'V':1,'F':0,'Y':-1,'W':-3}
		L={'C':-1,'S':-2,'T':-2,'P':-3,'A':-1,'G':-4,'N':-3,'D':-4,'E':-3,'Q':-2,'H':-3,'R':-2,'K':-2,'M':2,'I':2,'L':4,'V':3,'F':0,'Y':-1,'W':-2}
		V={'C':-1,'S':-2,'T':-2,'P':-2,'A':0,'G':-3,'N':-3,'D':-3,'E':-2,'Q':-2,'H':-3,'R':-3,'K':-2,'M':1,'I':3,'L':1,'V':4,'F':-1,'Y':-1,'W':-3}
		F={'C':-2,'S':-2,'T':-2,'P':-4,'A':-2,'G':-3,'N':-3,'D':-3,'E':-3,'Q':-3,'H':-1,'R':-3,'K':-3,'M':0,'I':0,'L':0,'V':-1,'F':6,'Y':3,'W':1}
		Y={'C':-2,'S':-2,'T':-2,'P':-3,'A':-2,'G':-3,'N':-2,'D':-3,'E':-2,'Q':-1,'H':2,'R':-2,'K':-2,'M':-1,'I':-1,'L':-1,'V':-1,'F':3,'Y':left,'W':2}
		W={'C':-2,'S':-3,'T':-3,'P':-4,'A':-3,'G':-2,'N':-4,'D':-4,'E':-3,'Q':-2,'H':-2,'R':-3,'K':-3,'M':-1,'I':-3,'L':-2,'V':-3,'F':1,'Y':2,'W':11}
		
		blosum62={'C':C,'S':S,'T':T,'P':P,'A':A,'G':G,'N':N,'D':D,'E':E,'Q':Q,'H':H,'R':R,'K':K,'M':M,'I':I,'L':L,'V':V,'F':F,'Y':Y,'W':W}
		
		
######################将X取其他aa的均值加入到各个分字典中，另创建一个分字典并在总字典中添加X分字典################
		X={};maxi=0
		for x in blosum62.keys():
			he=0
			for m in blosum62[x].values():
				he+=m
			blosum62[x]['X']=he*1.0/20
			X[x]=he/20.0
			maxi+=X[x]
		X['X']=maxi*1.0/20	
		blosum62['X']=X
		print (blosum62.values());print (X)

#####################找出前10个neighbors，并求其中阳性数据的比例############################		
		KNN={}
		for x in all_dataset:
		#x='TMVKQMTDVLLTP'
			distance={};k_neighbors=[]
			for m in all_dataset:
				if m!=x:
					dist=0
					for i in range(13):
						dist+=blosum62[x[i]][m[i]]
					distance[m]=dist
			#print (distance)
			k=10
			sorted_distance=sorted(zip(distance.values(),distance.keys()))	#zip函数作用于可迭代对象，将对应的元素打包成一个个元组，然后返回有这些元素组成的列表，sort函数按照列表中元组的第一个元素的大小从小到大排序
			k_neighbors=sorted_distance[-k:]
			#print (k_neighbors)
					
			number=0
			for i in k_neighbors:
				if i[1] in T_positive:
					number+=1
			KNN[x]=number*1.0/k
		
		
		
		
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

		# for n in T_positive2:
			# vec=[]
			# for a in n:
				# for x in aa:
					# if a==x:
						# if x!='X':
							# vec.append(1)
						# else:
							# vec.append(0.5)
					# else:
						# vec.append(0)
			# #final_vec3.append(vec)
			# ####基于二级结构
			# for i in positive1[n]:
				# for x in structure:
					# if i==x:
						# if x!='X':
							# vec.append(1)
						# else:
							# vec.append(0.5)
					# else:
						# vec.append(0)
			# ####基于疏水性
			# for i in positive2[n]:
				# for x in solvent:
					# if i==x:
						# if x!='X':
							# vec.append(1)
						# else:
							# vec.append(0.5)
					# else:
						# vec.append(0)
			# ####基于有序性
			# for i in positive3[n]:
				# for x in order:
					# if i==x:
						# if x!='X':
							# vec.append(1)
						# else:
							# vec.append(0.5)
					# else:
						# vec.append(0)
			# ####基于KNN
			# vec.append(KNN[n])
			# final_vec1.append(vec)
		# print (len(final_vec1[1]))
		print (len(final_vec3[1]))
		
		
		for n in T_negative2:
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
			final_vec4.append(vec)
		
		
		# for n in T_negative2:
			# vec=[]
			# for a in n:
				# for x in aa:
					# if a==x:
						# if x!='X':
							# vec.append(1)
						# else:
							# vec.append(0.5)
					# else:
						# vec.append(0)
			# #final_vec4.append(vec)
			# ####基于二级结构
			# for i in all_dataset1[n]:
				# for x in structure:
					# if i==x:
						# if x!='X':
							# vec.append(1)
						# else:
							# vec.append(0.5)
					# else:
						# vec.append(0)
			# ####基于疏水性
			# for i in all_dataset2[n]:
				# for x in solvent:
					# if i==x:
						# if x!='X':
							# vec.append(1)
						# else:
							# vec.append(0.5)
					# else:
						# vec.append(0)
			# ####基于有序性
			# for i in all_dataset3[n]:
				# for x in order:
					# if i==x:
						# if x!='X':
							# vec.append(1)
						# else:
							# vec.append(0.5)
					# else:
						# vec.append(0)
			# ####基于KNN
			# vec.append(KNN[n])
			# final_vec2.append(vec)
		# print (final_vec2[1])
		print (final_vec4[1])
		
	
# ######################生成特征向量文件，用于mrmr		
		# with open(r'C:\Users\xzw\Desktop\CDK_feature.csv','w') as out_f:
			# for i in final_vec1:
				# out_f.write(','.join(i)+'\n')
			# for i in final_vec2:
				# out_f.write(','.join(i)+'\n')
				
				
				
				
		# mRMR_ATM=[160,145,416,356,162,163,141,164,354,15right,154,156,155,393,153,14left,161,30left,45,152,339,332,351,63,1right1,151,150,1left0,311,90,3left3,113,350,14right,246,142,15left,36left,366,69,35right,2left2,303,200,54,149,right0,215,1left2,43]
		# mRMR_CDK=[159,145,354,416,290,19left,14left,29left,162,96,303,156,152,293,203,356,161,164,29right,150,3left0,160,301,155,163,355,294,154,364,151,299,360,149,2leftright,342,166,34left,15right,15left,rightleft,19right,305,1right2,334,191,2right6,366,192,165,30left]
		# new_vec1=[];new_vec2=[]

		# for i in final_vec1:
			# vec=[]
			# for x in mRMR_CDK:
				# vec.append(i[x])
			# new_vec1.append(vec)
		
		# for i in final_vec2:
			# vec=[]
			# for x in mRMR_CDK:
				# vec.append(i[x])
			# new_vec2.append(vec)

		# print (new_vec1[5])

			
# #####################生成特征值文件用于随机森林模型#####################			
		# with open(r'C:\Users\YU\Desktop\ATM_RF.csv','w') as out_f:
			# for i in new_vec1:
				# out_f.write(','.join(i)+'\n')
			# for i in new_vec2:
				# out_f.write(','.join(i)+'\n')
						
		
	

		vec1 = np.array(final_vec3+ final_vec4)
		vec_label1 = np.array([1] * len(final_vec3) + [0] * len(final_vec4))

		
		# vec2 = np.array(final_vec3+ final_vec4)
		# vec_label2 = np.array([1] * len(final_vec3) + [0] * len(final_vec4))

##########svm###########
		clf=svm.LinearSVC()
		scores = model_selection.cross_val_score(clf, vec1, y=vec_label1, cv=5)
		print('Per accuracy in 5-fold CV:')
		print(scores)
		print("Accuracy of svm: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
		
		
#########mlp############
		# classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(175, 75), random_state=1)
		# scores = model_selection.cross_val_score(classifier, vec, y=vec_label, cv=5)
		# print('Per accuracy in 5-fold CV:')
		# print(scores)
		# print("Accuracy of mlp: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

		
# Classification and ROC analysis.

#Run classifier with cross-validation and plot ROC curves


# cv = StratifiedKFold(n_splits=10)
# classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(175, 75), random_state=1)


# mean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)
# all_tpr = []
# cnt=0
# for i, (train, test) in enumerate(cv.split(vec1,vec_label1)):
    # cnt=cnt+1
    # probas_ = classifier.fit(vec1[train], vec_label1[train]).predict_proba(vec1[test])
    # fpr, tpr, thresholds = roc_curve(vec_label1[test], probas_[:, 1])
    # mean_tpr += interp(mean_fpr, fpr, tpr)
    # mean_tpr[0] = 0.0

# # Plot ROC curve.
# mean_tpr /= cnt
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, '-', label='Mean ROC1 (area = %0.2f)' % mean_auc, lw=2)



# ##第二个ROC
# mean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)
# all_tpr = []
# cnt=0
# for i, (train, test) in enumerate(cv.split(vec2,vec_label2)):
    # cnt=cnt+1
    # probas_ = classifier.fit(vec2[train], vec_label2[train]).predict_proba(vec2[test])
    # fpr, tpr, thresholds = roc_curve(vec_label2[test], probas_[:, 1])
    # mean_tpr += interp(mean_fpr, fpr, tpr)
    # mean_tpr[0] = 0.0

# # Plot ROC curve.
# mean_tpr /= cnt
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, '-', label='Mean ROC2 (area = %0.2f)' % mean_auc, lw=2)

# plt.xlim([0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")
# plt.show()