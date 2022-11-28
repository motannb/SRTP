# *-* encoding:gbk
import numpy as np      #矩阵运算库
import matplotlib.pyplot as plt #绘图库
from mpl_toolkits.mplot3d import Axes3D #绘制三维图
import pandas as pd     #读取数据库
from sklearn.model_selection import train_test_split,cross_val_score    #划分训练集和测试集 交叉验证得分
from sklearn.preprocessing import LabelBinarizer        #独热编码
from sklearn.metrics import accuracy_score,confusion_matrix              #分类准确率 混淆矩阵
from sklearn.preprocessing import StandardScaler        #标准化
from sklearn.decomposition import PCA                   #PCA降维算法
"""
Author:motan
Date:2022.11.27
Content:tool-package for users
"""

color = ['black', 'grey', 'lightcoral', 'red', 'darkorange', 'burlywood', 'gold', 'forestgreen', 'slateblue',
         'lightseagreen', 'royalblue', 'navy', 'blueviolet', 'orchid', 'cyan', 'pink', 'honeydew']      #颜色包

def Visualization_3D(X_test,y_test,voltage_num,title,score,save_path):
    """
    该函数算法内部调用
    将分类结果以三维图形式可视化
    :param X_test: 自变量测试集
    :param y_test: 因变量测试集
    :param voltage_num: 电压种类数
    :param title: 标题
    :param score: 分类得分
    :param save_path: 保存路径 若使用PyCharm打开项目应保留.. 示例:"../Result/KNN/{}.png" 使用Vscode打开项目不保留.. 示例:"Result/KNN/{}.png"
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
    该函数算法内部调用
    将分类结果以二维图形式可视化
    :param X: 自变量测试集
    :param y: 因变量测试集
    :param n_components: 提取主成分数
    :param voltage_num:  电压种类数
    :param title: 标题
    :param score: 分类得分
    :param save_path: 保存路径 若使用PyCharm打开项目应保留.. 示例:"../Result/KNN/{}.png" 使用Vscode打开项目不保留.. 示例:"Result/KNN/{}.png"
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
    用户调用函数 自定义训练模型
    :param model: 分类训练模型
    :param X: 自变量
    :param y: 标签
    :param train_size:   训练集大小
    :param random_state: 随机数种子
    :param voltage_num:  电压种类数
    :param title: 标题
    :param dimensionality: 维度
    :param save_path: 保存路径 若使用PyCharm打开项目应保留.. 示例:"../Result/KNN/{}.png" 使用Vscode打开项目不保留.. 示例:"Result/KNN/{}.png"
    :return:
    """
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=train_size,random_state=random_state) #划分训练集和测试集
    clf=model
    clf.fit(X_train,y_train)        #带入训练集
    score=clf.score(X_test,y_test)  #评分
    cross_score=cross_val_score(clf,X_train,y_train,cv=5)
    y_pred=clf.predict(X_test)
    c=confusion_matrix(y_test,y_pred)
    print(c)
    print("\n")
    if(dimensionality=='2D'):
        PCA_Visualization_2D(X_test,y_test,2,voltage_num,title,cross_score.mean(),save_path)  #PCA降维可视化
    if(dimensionality=='3D'):
        Visualization_3D(X, y, voltage_num, title, cross_score.mean(),save_path)              #三维可视化

def txt_to_csv(file_path,save_path,sep):
    """
    将txt文本转为csv文件 使用示例:txt_to_csv("../Data/result.txt","../Data/result.csv",'        ')
    :param file_path: 文件路径
    :param save_path: 保存路径
    :param sep: 分隔符 如'\t' ',' ' '
    :return: None
    """
    df=pd.read_csv(file_path,sep=sep,engine='python')        #读取数据
    df.to_csv(save_path,encoding='utf-8',index=False)


