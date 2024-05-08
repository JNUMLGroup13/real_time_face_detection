import sklearn
import numpy as np
import keras

# 读取数据
try :
    data = np.loadtxt('/facial-keypoints-detection/training.csv', delimiter=',')
except:
    print('Error: 读取数据失败')

