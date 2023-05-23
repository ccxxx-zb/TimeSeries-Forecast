from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.size']=13
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#加载数据
df = pd.read_csv('./dataset/Amazon.csv')
data=df['Close']
train_size = int(len(data) * 0.8)
train, test = data[0:train_size].tolist(), data[train_size:].tolist()
a1=np.load('./结果/Arima_Stock.npy')
a1=a1.tolist()#[0:30]
a2=np.load('./结果/SVR_Stock.npy')
a2=a2.tolist()#[0:30]
a3=np.load('./结果/LSTM_Stock.npy0.9870029200907927.npy')
a3=a3.tolist()#[0:30]
a4=np.load('./结果/Trans_Stock.npy0.9871571637290777.npy')
a4=a4.tolist()#[0:30]
# plt.plot(test,label='真实值',marker = "o",linewidth=2)
# plt.plot(a1,label='ARIMA',marker = "x",linewidth=2)
# plt.plot(a2,label='SVR',marker = ".",linewidth=2)
# plt.plot(a3,label='LSTM',marker = "+",linewidth=2)
# plt.plot(a4,label='Transformer',marker = "*",linewidth=2)
# plt.legend()
# plt.show()
plt.plot(test,label='真实值')#,marker = "o",linewidth=2)
# plt.plot(a1,label='ARIMA')#,marker = "x",linewidth=2)
plt.plot(a2,label='SVR')#,marker = ".",linewidth=2)
plt.plot(a3,label='LSTM')#,marker = "+",linewidth=2)
plt.plot(a4,label='Transformer')#,marker = "*",linewidth=2)
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend()
plt.show()
plt.plot(test[0:30],label='真实值',marker = "o",linewidth=2)
plt.plot(a2[0:30],label='SVR',marker = ".",linewidth=2)
plt.plot(a3[0:30],label='LSTM',marker = "+",linewidth=2)
plt.plot(a4[0:30],label='Transformer',marker = "*",linewidth=2)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()
df = pd.read_csv('./dataset/ILI.csv')
data=df['ILITOTAL']
train_size = int(len(data) * 0.8)
train, test = data[0:train_size].tolist(), data[train_size:].tolist()
# a1=np.load('./结果/Arima_ILI_new.npy')
# a1=a1.tolist()#[0:30]
a2=np.load('./结果/SVR_ILI.npy')
a2=a2.tolist()#[0:30]
a3=np.load('./结果/LSTM_ILI.npy0.82024537209979.npy')
a3=a3.tolist()#[0:30]
a4=np.load('./结果/Trans_ILI.npy0.7715465046604957.npy')
a4=a4.tolist()#[0:30]
# plt.plot(test[0:30],label='真实值',marker = "o",linewidth=2)
# plt.plot(a2[0:30],label='SVR',marker = ".",linewidth=2)
# plt.plot(a3[0:30],label='LSTM',marker = "+",linewidth=2)
# plt.plot(a4[0:30],label='Transformer',marker = "*",linewidth=2)
# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('ILITOTAL')
# # plt.show()
# fig_name = 'C://Users//77495//Desktop//ill1.svg'
# plt.savefig(fig_name, bbox_inches='tight')
#
plt.plot(test,label='真实值')#,marker = "o",linewidth=2)
# plt.plot(a1,label='ARIMA')#,marker = "x",linewidth=2)
plt.plot(a2,label='SVR')#,marker = ".",linewidth=2)
plt.plot(a3,label='LSTM')#,marker = "+",linewidth=2)
plt.plot(a4,label='Transformer')#,marker = "*",linewidth=2)
plt.legend()
plt.xlabel('Date')
plt.ylabel('ILITOTAL')
# plt.show()
fig_name = 'C://Users//77495//Desktop//ill2.svg'
plt.savefig(fig_name, bbox_inches='tight')
# plt.plot(test,label='真实值',marker = "o",linewidth=2)
# plt.plot(a1,label='ARIMA',marker = "x",linewidth=2)
# plt.legend()
# plt.show()
