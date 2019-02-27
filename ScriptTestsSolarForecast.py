from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import random
import csv
import os

#Converte o conjunto original em um conjunto janelado
def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

#Sequencias de entrada (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

#Sequencia a ser prevista (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

#Concatena as sequencias
    agg = concat(cols, axis=1)
    agg.columns = names

#Remove colunas com valores NaN
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#Quantidade de neuronios na camada de saida e na escondida
n_neurons = 50
u_bias = False

#Proporcoes de teste
tests = [0.25, 0.33, 0.5, 0.67, 0.75]

n_in_set = [3,13,91]
n_out_set = [1,3,6,13]

#Carrega o conjunto de dados em memoria
year = 2012
salve = 'Alfredo'  # Alterar para o seu recorte
dataset = read_csv(salve + '/' + salve + str(year) + '.csv', header=0, index_col=0)

#laco de combinacoes de parametros
for var_set in ['A','B','C']:

	if var_set == 'B':
		dataset.drop(dataset.columns[[1, 2, 3, 5, 7, 9]], axis=1, inplace = True)	
	elif var_set == 'C':
		dataset.drop(dataset.columns[[2, 3, 5, 6, 9]], axis=1, inplace = True)

#Garante que todos os valores sao do tipo float
	values = dataset.values
	values = values.astype('float64')
	n_features = dataset.shape[1]

#Normaliza os valores entre 0 e 1
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

#Laco que especifica quantas horas serao previstas
	for n_hours in n_in_set:
		n_obs = n_hours * n_features

#Laco que especifica quantas
		for n_out in n_out_set:

			remove = []
			reframed = series_to_supervised(scaled, n_hours, n_out)
			
			for i in range(n_out):
				for j in range(n_features-1):
					remove.append(n_obs + 1 + i*n_features + j)

			reframed.drop(reframed.columns[remove], axis=1, inplace=True)
			#print(reframed.shape)
            
#Divide em treino e teste
			values = reframed.values
			X = values[:, :n_obs]
			y = values[:, -1]

			names = []
			RMSEs = []
			dir_name = var_set+'_'+salve+'_'+str(n_hours)+'_'+str(n_out) 
			os.mkdir(dir_name)


#laco que realiza o treino e teste e grava os resultados
			for test in tests:

#Divide os conjuntos de treino e teste
				test_name = dir_name+'/_'+str(test)
				os.mkdir(test_name)
				train_X, test_X, train_y, test_y = train_test_split(
				    X, y, test_size=test, shuffle=False, stratify=None)

				train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
				test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
				#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

				random.seed(69)

#Arquitetura da rede
				model = Sequential()
				model.add(LSTM(n_neurons, input_shape=(
				    train_X.shape[1], train_X.shape[2]), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=u_bias))
				model.add(Dense(1))
				model.compile(loss='mae', optimizer='adam')

#Treina o modelo
				history = model.fit(train_X, train_y, epochs=100, batch_size=75, validation_data=(
				    test_X, test_y), verbose=2, shuffle=False)

				pyplot.plot(history.history['loss'], label='train')
				pyplot.plot(history.history['val_loss'], label='test')
				pyplot.legend()
				pyplot.savefig(test_name+'/errorsXepochs.png')
				pyplot.gcf().clear()

#Realiza predicoes
				yhat = 0
				yhat = model.predict(test_X)
				type(yhat)
				test_X = test_X.reshape((test_X.shape[0], n_obs))
				# invert scaling for forecast
				inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
				inv_yhat = scaler.inverse_transform(inv_yhat)
				inv_yhat = inv_yhat[:, 0]
				# invert scaling for actual
				test_y = test_y.reshape((len(test_y), 1))
				inv_y = concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
				inv_y = scaler.inverse_transform(inv_y)
				inv_y = inv_y[:, 0]
#Calcula RMSE
				rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
				RMSEs.append(rmse)
				names.append(str(test))
				print('Test RMSE: %.3f' % rmse)

#Grava resultados
				with open(test_name+ '/inv_y.csv', 'w', newline='') as myfile:
					wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
					for i in range(len(inv_y)):
						elem = [inv_y[i]]
						wr.writerow(elem)
						elem = []

				with open(test_name+'/inv_yhat.csv', 'w', newline='') as myfile:
					wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
					for i in range(len(inv_yhat)):
						elem = [inv_yhat[i]]
						wr.writerow(elem)
						elem = []

			with open(dir_name+'/erros.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
				wr.writerow(names)
				wr.writerow(RMSEs)
