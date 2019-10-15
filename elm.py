#coding=utf8

import re

# ##筛选出无重复的所有人类磷酸化底物蛋白
# with open(r'C:\Users\xzw\Desktop\human_elm.fasta')as f:
#     lst=f.readlines();cd_hit=[]
#     for i in lst:
#         seq='>'+i.split('\t')[0]+'\n'+i.split('\t')[1]
#         if seq not in cd_hit:
#             # cd_hit.append(i.split('\t')[0])
#             cd_hit.append(seq)
#     print len(cd_hit)
# f.close()
#
# with open(r'C:\Users\xzw\Desktop\human_redun_elm_.fasta','w')as f:
#     for i in cd_hit:
#         f.write(i+'\n')
# f.close()
#
#
# ##使用cd-hit去重70%，生成E:\python_project\original_data\human_elm_cd.fasta
#
#
# ##去除多余的换行符
# with open(r'E:\python_project\original_data\human_elm_cd.fasta')as f:
#     with open(r'E:\python_project\original_data\human_elm_cd2.fasta','w')as f2:
# 		lst=f.readlines();print len(lst)
# 		for i in lst:
# 			if len(i)>2:
# 				f2.write(i)
# 	f2.close()
# f.close()
#
# ##查看一个换行符的长度
# ha='\n'
# print len(cha)

def extract_specific_site(*kw):
    with open(kw[0])as f:
        with open(kw[1])as f2:
            lst1=f.readlines();lst2=f2.readlines();term=[];substrate=[]
            for i in [x.strip('\n')[1:] for x in lst1 if '>' in x]:
                    for x in lst2:
                        if i==x.split('\t')[0] and x.split('\t')[5]:
                            term.append(x)
                            if i not in substrate:
                                substrate.append(i)
                                
        f2.close()
    f.close()
    print len(term),len(substrate)
    
    kinases = [];sub = []
    for i in term:
        if i.split('\t')[5] not in kinases:
            kinases.append(i.split('\t')[5])
        if re.findall(kw[2],i.split('\t')[5]):
            sub.append(i)
    print len(kinases);print kinases;print len(sub)
    
    

if __name__=='__main__':
    extract_specific_site(r'E:\python_project\original_data\human_elm_cd2.fasta',r'E:\python_project\original_data\human_elm.fasta','^PKC')