# *-* encoding:gbk
import numpy as np      #���������
import matplotlib.pyplot as plt #��ͼ��
import pandas as pd     #��ȡ���ݿ�
from sklearn.model_selection import train_test_split,cross_val_score    #����ѵ�����Ͳ��Լ�
from sklearn.preprocessing import LabelBinarizer        #���ȱ���
from sklearn.metrics import accuracy_score,confusion_matrix              #����׼ȷ��
from sklearn.cross_decomposition import PLSRegression   #PLS�㷨
from sklearn.preprocessing import StandardScaler        #��׼��
"""
Author:motan
Date:2022.11.27
Content:PLS-DA
"""

if __name__=='__main__':
    data=np.loadtxt("../Data/result.txt",dtype=np.str_,encoding='utf-8')       #��ȡ����
    df=pd.read_csv("../Data/result.txt",sep='        ',engine='python')        #��ȡ����
    df.columns=['Type','Voltage','L','a','b']
    color_type=list(set(df['Type']))        #��ɫ����
    color = data[:, 0]
    for i in color_type:
        X=data[color==i,2:].astype(float)       #LABֵ
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
        print('���Լ���������Ϊ��\n', confusion_matrix(y_test, y_pred))
        print('ƽ������׼ȷ��Ϊ��\n', accuracy_score(y_test, y_pred))