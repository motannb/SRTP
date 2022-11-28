# *-* encoding:gbk
import numpy as np      #���������
import matplotlib.pyplot as plt #��ͼ��
from mpl_toolkits.mplot3d import Axes3D #������άͼ
import pandas as pd     #��ȡ���ݿ�
from sklearn.model_selection import train_test_split,cross_val_score    #����ѵ�����Ͳ��Լ� ������֤�÷�
from sklearn.preprocessing import LabelBinarizer        #���ȱ���
from sklearn.metrics import accuracy_score,confusion_matrix              #����׼ȷ�� ��������
from sklearn.preprocessing import StandardScaler        #��׼��
from sklearn.decomposition import PCA                   #PCA��ά�㷨
"""
Author:motan
Date:2022.11.27
Content:tool-package for users
"""

color = ['black', 'grey', 'lightcoral', 'red', 'darkorange', 'burlywood', 'gold', 'forestgreen', 'slateblue',
         'lightseagreen', 'royalblue', 'navy', 'blueviolet', 'orchid', 'cyan', 'pink', 'honeydew']      #��ɫ��

def Visualization_3D(X_test,y_test,voltage_num,title,score,save_path):
    """
    �ú����㷨�ڲ�����
    ������������άͼ��ʽ���ӻ�
    :param X_test: �Ա������Լ�
    :param y_test: ��������Լ�
    :param voltage_num: ��ѹ������
    :param title: ����
    :param score: ����÷�
    :param save_path: ����·�� ��ʹ��PyCharm����ĿӦ����.. ʾ��:"../Result/KNN/{}.png" ʹ��Vscode����Ŀ������.. ʾ��:"Result/KNN/{}.png"
    :return: None
    """
    fig=plt.figure()
    ax=Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    for i in range(0,voltage_num):
        ax.scatter(X_test[y_test==i,0],X_test[y_test==i,1],X_test[y_test==i,2],c=color[i],label=i*5)
    ax.set_xlabel('L')
    ax.set_ylabel('a')
    ax.set_zlabel('b')
    ax.text2D(0.25, 0.95, "{} recognition result of voltage Score:{:.3}".format(title,score), transform=ax.transAxes)
    ax.legend(bbox_to_anchor=(0, 1),title='voltage')
    plt.savefig(save_path)
    plt.show()

def PCA_Visualization_2D(X,y,n_components,voltage_num,title,score,save_path):
    """
    �ú����㷨�ڲ�����
    ���������Զ�άͼ��ʽ���ӻ�
    :param X: �Ա������Լ�
    :param y: ��������Լ�
    :param n_components: ��ȡ���ɷ���
    :param voltage_num:  ��ѹ������
    :param title: ����
    :param score: ����÷�
    :param save_path: ����·�� ��ʹ��PyCharm����ĿӦ����.. ʾ��:"../Result/KNN/{}.png" ʹ��Vscode����Ŀ������.. ʾ��:"Result/KNN/{}.png"
    :return: None
    """
    pca=PCA(n_components=n_components)
    X=pca.fit_transform(X)
    print(X.shape,y.shape)
    plt.figure()
    for i in range(0,voltage_num):
        plt.scatter(X[y==i,0],X[y==i,1],c=color[i],label=i*5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(bbox_to_anchor=(-0.15, 1),title='voltage')
    plt.title("{} recognition result of voltage Score:{:.3}".format(title,score))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def Pattern_Recognition(model,X,y,train_size,random_state,voltage_num,title,dimensionality,save_path):
    """
    �û����ú��� �Զ���ѵ��ģ��
    :param model: ����ѵ��ģ��
    :param X: �Ա���
    :param y: ��ǩ
    :param train_size:   ѵ������С
    :param random_state: ���������
    :param voltage_num:  ��ѹ������
    :param title: ����
    :param dimensionality: ά��
    :param save_path: ����·�� ��ʹ��PyCharm����ĿӦ����.. ʾ��:"../Result/KNN/{}.png" ʹ��Vscode����Ŀ������.. ʾ��:"Result/KNN/{}.png"
    :return:
    """
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=train_size,random_state=random_state) #����ѵ�����Ͳ��Լ�
    clf=model
    clf.fit(X_train,y_train)        #����ѵ����
    score=clf.score(X_test,y_test)  #����
    cross_score=cross_val_score(clf,X_train,y_train,cv=5)
    y_pred=clf.predict(X_test)
    c=confusion_matrix(y_test,y_pred)
    print(c)
    print("\n")
    if(dimensionality=='2D'):
        PCA_Visualization_2D(X_test,y_test,2,voltage_num,title,cross_score.mean(),save_path)  #PCA��ά���ӻ�
    if(dimensionality=='3D'):
        Visualization_3D(X, y, voltage_num, title, cross_score.mean(),save_path)              #��ά���ӻ�

def txt_to_csv(file_path,save_path,sep):
    """
    ��txt�ı�תΪcsv�ļ� ʹ��ʾ��:txt_to_csv("../Data/result.txt","../Data/result.csv",'        ')
    :param file_path: �ļ�·��
    :param save_path: ����·��
    :param sep: �ָ��� ��'\t' ',' ' '
    :return: None
    """
    df=pd.read_csv(file_path,sep=sep,engine='python')        #��ȡ����
    df.to_csv(save_path,encoding='utf-8',index=False)


