#数据处理 

# 导入数据库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据集
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values
'''因为这里X、y中含有字符串，是object类型，无法直接在explorer中查看，可以在console中查看'''

# 缺失数据处理
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
''''这里可选择的strategy有平均数，众数，中位数，此处所有Imputer参数均为默认'''
X[:,1:3] = imputer.fit_transform(X[:, 1:3])

# 虚拟编码
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
'''LabelEncoder是将标签分类编码，此时仍是object类型
   OneHotEncoder是将分类结果用0-1矩阵表示，此时toarry转化为float or int
   categorical_features是分类结果所在的列,转化结果自动到首列
'''
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
'''为了避免虚拟陷阱，即信息重复，一般会将多余的列去除，这里的第一列是多余的，后两列足够表示
   国籍不同这一信息了
'''
X = X[:,1:] 

# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据划分训练集，测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


