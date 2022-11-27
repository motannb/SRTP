# *-* encoding:gbk
import numpy as np      #矩阵运算库
import matplotlib.pyplot as plt #绘图库
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd     #读取数据库
from sklearn.model_selection import train_test_split,cross_val_score    #划分训练集和测试集
from sklearn.preprocessing import LabelBinarizer        #独热编码
from sklearn.metrics import accuracy_score,confusion_matrix              #计算准确率
from sklearn.preprocessing import StandardScaler        #标准化
from sklearn.svm import SVC
from ML_tool_package import Pattern_Recognition
from sklearn.decomposition import PCA                   #PCA降维算法
"""
Author:motan
Date:2022.11.27
Content:SVM
"""

if __name__=='__main__':
    data=np.loadtxt("../Data/result.txt",dtype=np.str_,encoding='utf-8')       #读取数据
    df=pd.read_csv("../Data/result.txt",sep='        ',engine='python')        #读取数据
    df.columns=['Type','Voltage','L','a','b']
    color_type=list(set(df['Type']))        #颜色种类
    print(color_type)
    color=data[:,0]

    svm=SVC(kernel='rbf')   #实例化分类算法 此处设置参数仅用于测试 非最佳参数

    for i in color_type:
        X=data[color==i,2:].astype(float)       #LAB值
        pca = PCA(n_components=3)
        X = pca.fit_transform(X)
        scaler=StandardScaler()
        X=scaler.fit_transform(X)
        y = data[color == i, 1].astype(float)/5
        voltage_num=len(np.unique(y))
        Pattern_Recognition(svm,X,y,0.7,0,voltage_num,i,"3D","../Result/SVM/{}.png".format(i))
