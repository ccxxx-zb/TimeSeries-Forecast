import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.ticker as ticker
import statsmodels.tsa.stattools as st
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from statsmodels.tsa import stattools

plt.rcParams['font.size']=16
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


dataset = pd.read_csv(r'./dataset/Amazon.csv',encoding="utf-8")
#需要读取的数据
data_nodiff=pd.DataFrame(dataset['Close'].values,index=dataset['Date'])

#数据检测
# adftest = adfuller(data)
# print(adftest)
# plot_acf(data)
# plt.show()
# plot_pacf(data)
# plt.show()
# 非平稳转为平稳
data = data_nodiff.diff(1).dropna()
# predict=pd.DataFrame({0:predictions})
# predict.index=test.index
#再次检测
# adftest = adfuller(data)
# print(adftest)
# plot_acf(data)
# plt.show()
# plot_pacf(data)
# plt.show()
#白噪声检验
#res = acorr_ljungbox(data, lags=24, boxpierce=True, return_df=True)
#print(res)
train_size = int(len(data) * 0.8)
train, test = data[0:train_size][0].tolist(), data[train_size:][0].tolist()
#自动定阶1：一般选择BIC较小的值
# order=st.arma_order_select_ic(data,max_ar=3,max_ma=3,ic=['aic','bic','hqic'])
# print(order.bic_min_order)
# trend_evaluate = sm.tsa.arma_order_select_ic(data, ic=['aic', 'bic','hqic'], trend='n', max_ar=5,max_ma=5)
# print('train AIC', trend_evaluate.aic_min_order)
# print('train BIC', trend_evaluate.bic_min_order)
# print('train HQIC', trend_evaluate.hqic_min_order)
#自动定阶2：
#model_fit = auto_arima(train)#, start_p=1, start_q=1, max_p=5, max_q=5, m=12, start_P=0, seasonal=True, d=1, D=1,trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
#训练1：
model = sm.tsa.ARIMA(train, order=(2,4,2)) # p,d,q
result = model.fit()
print(result.summary())
predictions=result.forecast(len(test))
plt.plot(test, label='真实值')
plt.plot(predictions, label='预测值')
plt.legend(fontsize=16)
plt.show()
# 预测2：
# for t in range(15):#len(test)):
#     model_fit = auto_arima(train)
#     model_fit.fit(train)
#     yhat = model_fit.predict()[0]
#     # model = sm.tsa.ARIMA(train, order=(5, 1, 0))
# 	# result = model.fit()
#     # print(result.summary())
#     # yhat=result.forecast(1)
#     predictions.append(yhat)
#     train.append(yhat)
#     train=train[1:]
#数据还原
diff_shift_ts=data_nodiff.shift(1).dropna()
add=diff_shift_ts[-len(test):][0].tolist()
y_predict_unscale=[predictions[i]+add[i] for i in range(len(test))]
y_test_unscale=data_nodiff[-len(test):][0].tolist()
y_test = y_test_unscale
y_predict=y_predict_unscale

# 一些图+数据值
fig=plt.figure()
plt.plot(y_test, label='真实值')
plt.plot(y_predict, label='预测值')
plt.legend(fontsize=16)
plt.show()
print("mae =",mean_absolute_error(y_test, y_predict))
print("mse =", mean_squared_error(y_test, y_predict))
print("rmse =", np.sqrt(mean_squared_error(y_test, y_predict)))
print("r_2 =" ,r2_score(y_test, y_predict))
fig_name = 'ARIMA-Stock' + str(r2_score(y_test, y_predict)) + '.svg'
fig.savefig(fig_name,bbox_inches='tight')
pred_s=np.array(y_predict_unscale)
np.save('结果/Arima_Stock_new2.npy', pred_s)



