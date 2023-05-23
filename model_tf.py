import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Dense, Lambda
from tensorflow.keras.layers import Conv1D, UpSampling1D
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.layers import  Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow.keras.utils import plot_model

np.random.seed(1234)
tf.compat.v1.random.set_random_seed(1234)

# 损失函数绘制
def plotLosses(loss, name=[]):
	fig = plt.figure(figsize=(6, 4))
	plt.plot(loss[:, 0], label='loss', c='green')
	plt.legend()

	plt.grid(False)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')

	fig.tight_layout()
	fig.savefig(name + '.png')
	plt.close(fig)

#求解RMSE
def RMSE(x, y):
	return np.sqrt(np.mean(np.square(x.flatten() - y.flatten())))

class TF:

	def __init__(self, data, n_lag, n_seq, name=[]):
    
		self.name = name

		self.data = data
		self.n_lag = n_lag
		self.n_seq = n_seq

		self.n_batch = 0
		self.nb_epoch = 0

		self.x = []
		self.y = []
        
		self.model = []
		#一些参数：
		self.head_size = 3
		self.num_heads = 8
		self.ff_dim = 8
		self.num_transformer_blocks = 1
		self.mlp_units = [32,16]
		self.dropout = 0.4
		self.mlp_dropout = 0.25

	# 编码器
	def get_tf_encoder(self, input, head_size, num_heads, ff_dim, dropout=0):
		_ = LayerNormalization(epsilon=1e-6)(input)
		_ = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(_, _)
		_ = Dropout(dropout)(_)
		res = _ + input

		_ = LayerNormalization(epsilon=1e-6)(res)
		_ = Conv1D(filters=ff_dim, kernel_size=3, activation="relu", padding="same")(_)
		_ = Dropout(dropout)(_)
		_ = Conv1D(filters=input.shape[-1], kernel_size=1)(_)
		return _ + res
	# 整体架构
	def get_tf_model(self, input, output):
	
		input_data = Input(batch_shape=(self.n_batch, input.shape[1], input.shape[2]))
		_ = input_data
		for n in range(self.num_transformer_blocks):
			_ = self.get_tf_encoder(_, self.head_size, self.num_heads, self.ff_dim, self.dropout)
		_ = GlobalAveragePooling1D(data_format="channels_first")(_)
		for d in self.mlp_units:
			_ = Dense(d, activation="relu")(_)
			_ = Dropout(self.mlp_dropout)(_)
		output_data = Dense(output.shape[1])(_)

		self.model = Model(input_data, output_data)
		self.model.compile(loss='mse', optimizer=Adam(lr=1e-4))
		self.model.summary()
		plot_model(self.model, to_file='tf.png')
    # 模型训练
	def train(self, n_batch=1, nb_epoch=100, load=False):
    
		self.n_batch = n_batch
		self.nb_epoch = nb_epoch
		a=self.data.iloc[:, 0:self.n_lag]
		x = self.data.iloc[:, 0:self.n_lag].values
		self.x = x.reshape(x.shape[0], x.shape[1], 1)
		self.y = self.data.iloc[:, self.n_lag:].values

		self.get_tf_model(self.x, self.y)

		history = History()

		losses = np.zeros([self.nb_epoch, 2])

		for i in tqdm(range(self.nb_epoch)):
			self.model.fit(self.x, self.y, 
						epochs=1, batch_size=self.n_batch, verbose=False,
						shuffle=False, callbacks=[history,EarlyStopping(monitor='loss', patience=10, verbose=1)])
			self.model.reset_states()
			
			losses[i, :]  = np.asarray(list(history.history.values()))[:, i]
			print ("%d [Loss: %f] [Val loss: %f]" % (i, losses[i, 0], losses[i, 1]))
			
			figs = plotLosses(losses, name="tf_losses")
    # 模型预测
	def forecasts(self, test_data):
    
		forecasts = list()
		print('----------Predicting-------------')
		for i in range(len(test_data)):

			x = test_data.iloc[i, 0:self.n_lag].values
			x = x.reshape(1, len(x), 1)
			y = test_data.iloc[i, self.n_lag:].values
			
			y_hat = self.model.predict(x, batch_size=self.n_batch,verbose=0)
			forecasts.append([x for x in y_hat[0, :]])
		print('----------Over-------------')
		return forecasts
