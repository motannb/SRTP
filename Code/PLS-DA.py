# *-* encoding:gbk
import numpy as np      #矩阵运算库
import matplotlib.pyplot as plt #绘图库
import pandas as pd     #读取数据库
from sklearn.model_selection import train_test_split,cross_val_score    #划分训练集和测试集
from sklearn.preprocessing import LabelBinarizer        #独热编码
from sklearn.metrics import accuracy_score,confusion_matrix              #计算准确率
from sklearn.cross_decomposition import PLSRegression   #PLS算法
from sklearn.preprocessing import StandardScaler        #标准化
"""
Author:motan
Date:2022.11.27
Content:PLS-DA
"""

if __name__=='__main__':
    data=np.loadtxt("../Data/result.txt",dtype=np.str_,encoding='utf-8')       #读取数据
    df=pd.read_csv("../Data/result.txt",sep='        ',engine='python')        #读取数据
    df.columns=['Type','Voltage','L','a','b']
    color_type=list(set(df['Type']))        #颜色种类
    color = data[:, 0]
    for i in color_type:
        X=data[color==i,2:].astype(float)       #LAB值
        scaler=StandardScaler()
        X=scaler.fit_transform(X)
        y = data[color == i, 1].astype(float)/5
        voltage_num=len(np.unique(y))
        one_hot = LabelBinarizer()
        y = one_hot.fit_transform(y)
        X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=0)
        PLS=PLSRegression(n_components=2)
        PLS.fit(X_train,y_train)
        y_pred=PLS.predict(X_test)
        y_pred=one_hot.inverse_transform(y_pred)
        y_test=one_hot.inverse_transform(y_test)
        print('测试集混淆矩阵为：\n', confusion_matrix(y_test, y_pred))
        print('平均分类准确率为：\n', accuracy_score(y_test, y_pred))