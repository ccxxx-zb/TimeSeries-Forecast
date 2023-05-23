import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import model_tf
import pickle
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
plt.rcParams['font.size']=16
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(1234)


def series_to_supervised(data, n_lag, n_seq, dropnan=True):
    n_vars = 1
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_lag, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    for i in range(0, n_seq):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg

n_lag = 2
n_seq = 2
simulation=5

# 数据处理
df = pd.read_csv('./dataset/ILI.csv')
res=df['ILITOTAL'].tolist()
min_=np.min(res)
max_=np.max(res)
data_scale = (res -min_) / (max_- min_)
data=series_to_supervised(data_scale, n_lag, n_seq, dropnan=True)

# 数据划分
split=0.8
split_idx = int(data.shape[0] * split)
y0_train, y0_test = data[0:split_idx], data[split_idx:]
# print(y0_train.shape)
# print(y0_test.shape)

result=[]
result_unscale=[]
mae=[]
mse=[]
rmse=[]
mape=[]
r_2=[]

y_test = y0_test.iloc[:, -1].to_numpy()
y_test_unscale = (y_test * (max_ - min_)) + min_
#模型创建
model = model_tf.TF(y0_train, n_lag=n_lag, n_seq=n_seq, name="tf")
for i in range(simulation):
    #模型训练
    model.train(n_batch=8, nb_epoch=2, load=False)
    #结果预测
    forecasts_test = model.forecasts(y0_test)
    y_pred = np.array(forecasts_test)[:, -1:]
    y_pred_unscale = (y_pred * (max_ - min_)) + min_
    result.append(y_pred)
    result_unscale.append(y_pred_unscale)
    mae.append([mean_absolute_error(y_test, y_pred), mean_absolute_error(y_test_unscale, y_pred_unscale)])
    mse.append([mean_squared_error(y_test, y_pred), mean_squared_error(y_test_unscale, y_pred_unscale)])
    rmse.append([np.sqrt(mean_squared_error(y_test, y_pred)), np.sqrt(mean_squared_error(y_test_unscale, y_pred_unscale))])
    mape.append([(abs(y_pred - y_test) / y_test_unscale).mean(),(abs(y_pred_unscale - y_test_unscale) / y_test_unscale).mean()])
    r_2.append([r2_score(y_test, y_pred), r2_score(y_test_unscale, y_pred_unscale)])
    # 模型保存
    model_name = './model/Transformer-I' + str([i][0])
    pickle.dump(model, open(model_name, 'wb+'))
    # 数据保存
    pred_s = np.array(y_pred_unscale)
    name1 = './结果/Trans_ILI.npy' + str(r_2[i][0])
    np.save(name1, pred_s)
    fig = plt.figure()
    plt.plot(y_test_unscale, label='真实值')
    plt.plot(y_pred_unscale, label='预测值')
    plt.legend(fontsize=16)
    plt.show()
    fig_name = 'Transform+ILI（单）' + str(r_2[i][0]) + '.svg'
    fig.savefig(fig_name,bbox_inches='tight')

#结果分析
fig=plt.figure()
plt.plot(y_test_unscale,label='真实值')
for r in range(simulation):
    label_name='第'+str(r+1)+'次预测值'
    plt.plot(result_unscale[r],label=label_name)
plt.legend(fontsize=12)
# plt.show()
fig.savefig('Transformer-ILI' + '.svg',bbox_inches='tight')
print("mae =",mae)
print("mse =", mse)
print("rmse =", rmse)
print("mape =" ,mape)
print("r_2 =" ,r_2)
