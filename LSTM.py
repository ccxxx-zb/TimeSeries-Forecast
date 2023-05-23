import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import tensorflow as tf
import keras.backend as K
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM

plt.rcParams['font.size']=16
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(1234)
tf.compat.v1.random.set_random_seed(1234)

result=[]
result_unscale=[]
mae=[]
mse=[]
rmse=[]
mape=[]
r_2=[]

simulation=5
# 数据准备
df = pd.read_csv('./dataset/Amazon.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_idx = df.set_index(["Date"], drop=True) # Data成为索引，而不是列
data = df_idx[['Close']]
split=0.8
split_idx = int(data.shape[0] * split)
train, test = data[0:split_idx], data[split_idx:]
scale = MinMaxScaler()
train_scale = scale.fit_transform(train) #scale在训练集上训练并应用
test_scale = scale.transform(test)  # scale应用在测试集上。


# 窗口延迟处理
N=3
train_sc_df = pd.DataFrame(train_scale, columns=['Close'], index=train.index)
test_sc_df = pd.DataFrame(test_scale, columns=['Close'], index=test.index)
train_test_sc_df= pd.concat([train_sc_df, test_sc_df],axis=0)

#X=窗口大小，  Y=下一日Price
for s in range(1,N+1):
    train_sc_df['Price_X_{}'.format(s)] = train_sc_df['Close'].shift(s)
train_sc_df=train_sc_df[N-1:]

for s in range(1,N+1):
    train_test_sc_df['Price_X_{}'.format(s)] = train_test_sc_df['Close'].shift(s)
test_sc_df=train_test_sc_df[-len(test_sc_df):]


X_train = train_sc_df.iloc[:,1:]
Y_train = train_sc_df.iloc[:,[0]]
X_test = test_sc_df.iloc[:,1:]
Y_test = test_sc_df.iloc[:,[0]]
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values
X_train[np.isnan(X_train)] = 0
# print('X_Train size: ',np.shape(X_train))
# print('Y_Train size: ',np.shape(Y_train))


# 准备LSTM的数据（samples, time-lag, features)
X_tr_t = X_train.reshape(X_train.shape[0], N, 1)
X_tst_t = X_test.reshape(X_test.shape[0], N, 1)

# 构建LSTM模型
K.clear_session()
model_lstm = Sequential()
model_lstm.add(LSTM(units=100, input_shape=(N, 1), return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1))
model_lstm.add(Activation("relu"))
model_lstm.compile(loss="mse", optimizer="adam", metrics=['mae', 'mape'])
model_lstm.summary()
early_stop = EarlyStopping(monitor='loss', patience=20, verbose=1)# 早停
y_test=Y_test
y_test_unscale=scale.inverse_transform(Y_test)

for i in range(simulation):
    # 模型训练
    history_model_lstm = model_lstm.fit(X_tr_t, Y_train, epochs=200, batch_size=64, verbose=1, shuffle=False, callbacks=[early_stop])
    # 预测结果
    y_pred = model_lstm.predict(X_tst_t,verbose=1)
    # 逆归一化预测结果
    y_pred_unscale = scale.inverse_transform(y_pred)
    result.append(y_pred)
    result_unscale.append(y_pred_unscale)
    mae.append([mean_absolute_error(y_test, y_pred), mean_absolute_error(y_test_unscale, y_pred_unscale)])
    mse.append([mean_squared_error(y_test, y_pred), mean_squared_error(y_test_unscale, y_pred_unscale)])
    rmse.append([np.sqrt(mean_squared_error(y_test, y_pred)),np.sqrt(mean_squared_error(y_test_unscale, y_pred_unscale))])
    mape.append([(abs(y_pred - y_test) / y_test_unscale).mean(),(abs(y_pred_unscale - y_test_unscale) / y_test_unscale).mean()])
    r_2.append([r2_score(y_test, y_pred), r2_score(y_test_unscale, y_pred_unscale)])
    # 模型保存
    model_name = './model/LSTM-S' + str(r_2[i][0])
    model_lstm.save(model_name)
    fig = plt.figure()
    plt.plot(y_test_unscale, label='真实值')
    plt.plot(y_pred_unscale, label='预测值')
    plt.legend(fontsize=16)
    fig_name = 'LSTM+Stock（单）' + str(r_2[i][0]) + '.svg'
    fig.savefig(fig_name, bbox_inches='tight')
    fig = plt.figure()
    plt.plot(history_model_lstm.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.show()
    fig_name = 'LSTM+Stock（单）+loss' + str(r_2[i][0])+'.svg'
    fig.savefig(fig_name, bbox_inches='tight')
    pred_s = np.array(y_pred_unscale)
    name1='./结果/LSTM_Stock.npy'+str(r_2[i][0])
    np.save(name1, pred_s)

#重复实验
fig=plt.figure()
plt.plot(y_test_unscale,label='真实值')
for r in range(simulation):
    label_name='第'+str(r+1)+'次预测值'
    plt.plot(result_unscale[r],label=label_name)
plt.legend(fontsize=12)
# plt.show()
fig.savefig('LSTM-Stock.svg',bbox_inches='tight')
#数值分析
print("mae =",mae)
print("mse =", mse)
print("rmse =", rmse)
print("mape =" ,mape)
print("r_2 =" ,r_2)

