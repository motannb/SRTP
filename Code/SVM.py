# *-* encoding:gbk
import numpy as np      #���������
import matplotlib.pyplot as plt #��ͼ��
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd     #��ȡ���ݿ�
from sklearn.model_selection import train_test_split,cross_val_score    #����ѵ�����Ͳ��Լ�
from sklearn.preprocessing import LabelBinarizer        #���ȱ���
from sklearn.metrics import accuracy_score,confusion_matrix              #����׼ȷ��
from sklearn.preprocessing import StandardScaler        #��׼��
from sklearn.svm import SVC
from ML_tool_package import Pattern_Recognition
from sklearn.decomposition import PCA                   #PCA��ά�㷨
"""
Author:motan
Date:2022.11.27
Content:SVM
"""

if __name__=='__main__':
    data=np.loadtxt("../Data/result.txt",dtype=np.str_,encoding='utf-8')       #��ȡ����
    df=pd.read_csv("../Data/result.txt",sep='        ',engine='python')        #��ȡ����
    df.columns=['Type','Voltage','L','a','b']
    color_type=list(set(df['Type']))        #��ɫ����
    print(color_type)
    color=data[:,0]

    svm=SVC(kernel='rbf')   #ʵ���������㷨 �˴����ò��������ڲ��� ����Ѳ���

    for i in color_type:
        X=data[color==i,2:].astype(float)       #LABֵ
        pca = PCA(n_components=3)
        X = pca.fit_transform(X)
        scaler=StandardScaler()
        X=scaler.fit_transform(X)
        y = data[color == i, 1].astype(float)/5
        voltage_num=len(np.unique(y))
        Pattern_Recognition(svm,X,y,0.7,0,voltage_num,i,"3D","../Result/SVM/{}.png".format(i))
