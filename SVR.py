import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import pickle
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

plt.rcParams['font.size']=16
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(1234)

result=[]
result_unscale=[]
mae=[]
mse=[]
rmse=[]
mape=[]
r_2=[]

#需要/可以修改的数据
df = pd.read_csv('./dataset/Amazon.csv')
df.head()
minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32'))
df_log = minmax.transform(df.iloc[:, 4:5].astype('float32'))
df_log = pd.DataFrame(df_log)
df_log.head()
x_data = df['Date']
y_data = pd.Series(df_log[0].values)

#预测的长度
test_size = 1100
simulation = 1
# 取前多少个X_data预测下一个数据
X_long = 3
#模型：kernel=rbf/linear/poly，C，gamma
model=SVR(kernel="poly", degree=2,C=100, epsilon=0.1,gamma="scale")

error = []
X = []
Y = []
long = len(x_data)
for k in range(len(x_data) - X_long - 1):
    t = k + X_long
    X.append(y_data[k:t])
    Y.append(y_data[t + 1])

# 数据划分
split=0.8
split_idx = int(len(X) * split)
x_train,x_test=X[0:split_idx],X[split_idx:]
y_train,y_test=Y[0:split_idx],Y[split_idx:]
y_test_unscale=minmax.inverse_transform(pd.DataFrame(y_test).values.reshape(1,-1)).reshape(-1,1)

# 随即搜索
# pipe_svr = Pipeline([("StandardScaler", StandardScaler()),
#                      ("svr", SVR())])

# distributions = dict(svr__C=uniform(loc=1.0, scale=4),
#                      svr__kernel=["linear", "rbf"],
#                      svr__gamma=uniform(loc=0, scale=4))
#
# rs = RandomizedSearchCV(estimator=pipe_svr,
#                         param_distributions=distributions,
#                         scoring='r2',
#                         cv=5)  # 10折交叉验证
# rs = rs.fit(x_train, y_train)


# 模型训练
train=model.fit(x_train,y_train)
# 预测结果
y_pred = train.predict(x_test)
# 逆归一化预测结果
y_pred_unscale = minmax.inverse_transform(pd.DataFrame(y_pred).values.reshape(1,-1)).reshape(-1,1)
result.append(y_pred)
result_unscale.append(y_pred_unscale)
mae.append([mean_absolute_error(y_test, y_pred), mean_absolute_error(y_test_unscale, y_pred_unscale)])
mse.append([mean_squared_error(y_test, y_pred), mean_squared_error(y_test_unscale, y_pred_unscale)])
rmse.append([np.sqrt(mean_squared_error(y_test, y_pred)),np.sqrt(mean_squared_error(y_test_unscale, y_pred_unscale))])
mape.append([(abs(y_pred - y_test) / y_test).mean(),(abs(y_pred_unscale - y_test_unscale) / y_test_unscale).mean()])
r_2.append([r2_score(y_test, y_pred), r2_score(y_test, y_pred)])
# 模型保存
model_name = './model/SVR-S_best'
pickle.dump(model, open(model_name, 'wb+'))
pred_s = np.array(y_pred_unscale)
np.save('./结果/SVR_Stock.npy', pred_s)
fig = plt.figure()
plt.plot(y_test_unscale, label='真实值')
plt.plot(y_pred_unscale, label='预测值')
plt.legend(fontsize=16)
fig_name = 'SVR+Stock（单）' + str(r_2[i][0]) + '.svg'
fig.savefig(fig_name, bbox_inches='tight')

print("mae =",mae)
print("mse =", mse)
print("rmse =", rmse)
print("mape =" ,mape)
print("r_2 =" ,r_2)

# # 交叉验证
# tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, test_size=None)
# rmse_list = []
# for kfold, (train_index, test_index) in enumerate(tscv.split(X)):
#     #print('train_index', train_index, 'test_index', test_index)
#     # 根据索引得到对应的训练集和测试集
#     train_X, train_y = x_train[train_index[1]:train_index[-1]], y_train[train_index[1]:train_index[-1]]
#     test_X, test_y = x_train[test_index[1]:test_index[-1]], y_train[test_index[1]:test_index[-1]]
#     # 建立模型并训练
#     model_k=model.fit(train_X, train_y)
#     # 计算测试集误差
#     test_pred = model_k.predict(test_X)
#     test_y = minmax.inverse_transform(pd.DataFrame(test_y).values.reshape(1, -1)).reshape(-1, 1)
#     test_pred=minmax.inverse_transform(pd.DataFrame(test_pred).values.reshape(1,-1)).reshape(-1,1)
#     plt.plot(test_y)
#     plt.plot(test_pred)
#     plt.show()
#     rmse = np.sqrt(np.mean(np.power((test_y - test_pred), 2)))
#     rmse_list.append(rmse)
#     # print('rmse of %d fold=%.4f' % (kfold, rmse))
# # 总的误差为每次误差的均值
# print('average rmse:', np.mean(rmse_list))







