# This file aims at using history electricity prices and carbon trading prices to predict short-term electricity prices
# The neural network being used is Recurrent Neural Network
# The input and ouput data should be placed in the same directory as the code, named input.csv and output.csv
# The data used is the N2EX Day Ahead Electricity Price in UK, and the EPEX Day Ahead Electricity Price in Germany

import numpy as np
from helper import stand_data, norm_data, norm_data_reverse
from keras.utils import plot_model
from keras.models import Sequential 
from keras.layers.recurrent import LSTM

def split_train_test(X, y, t, split_rate=0.1):  

    n_trn = int(X.shape[0]*(1-split_rate)) 
    t_train = t[:n_trn,:] 
    t_test = t[n_trn:,:]

    X_train = X[:n_trn,:]
    X_test = X[n_trn:,:]

    y_train = y[:n_trn]
    y_test = y[n_trn:]

    # report the dimensions of the train and test datasets
    print ('The dimensions for X_train is:', X_train.shape)
    print ('The dimensions for X_test is:', X_test.shape)
    print ('The dimensions for y_train dimensions is:', y_train.shape)
    print ('The dimensions for y_test dimensions is:', y_test.shape)
    print ('The dimensions for t_train is:', t_train.shape)
    print ('The dimensions for t_test is:', t_test.shape)
    
    return (X_train, y_train, t_train), (X_test, y_test, t_test)

def main():
    # read in the input and output data
    X = np.loadtxt('input0.csv', delimiter=",", skiprows=1)
    y = np.loadtxt('output2.csv', delimiter=",", skiprows=1)

    # eliminate the first column of input data, which is date
    t = X[:,0:1]
    X = X[:,1:] # ignore dates and hours in training

    # reshape y
    y = y.reshape(y.shape[0],1)

    # save output statistics for scaling back to absolute values
    y_max = y.max(axis=0)
    y_min = y.min(axis=0)

    # data standardization and normalization
    X = stand_data(X)
    y = norm_data(y, -1, 1)
    
    # split into train data sets and test sets
    (X_train, y_train, t_train), (X_test, y_test, t_test) = split_train_test(X, y, t)

    # reshape X_train and X_test for the need of LSTM networks
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # set the model epoch to be 500, which is enough for our task
    nb_epoch = 500
    batch_size = 200

    # Recurrent Neural Network model 
    model = Sequential()

    model.add(LSTM(50, input_dim=X_train.shape[2], return_sequences=True))
    model.add(LSTM(output_dim=1))

    model.summary()
    plot_model(model,to_file='model_diagram.png',show_shapes=True)

    # mean_squared_error and adam would be more suitable in this task
    model.compile(loss="mean_squared_error", optimizer='adam') 
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)

    # prediction
    prediction = model.predict(X_test, verbose=1)

    # use the min and max data to scale back to absolute values
    prediction = norm_data_reverse(prediction, -1, 1, y_min, y_max)
    y_test = norm_data_reverse(y_test, -1, 1, y_min, y_max)

    # output the results
    result = np.concatenate((t_test, prediction, y_test), axis=1)
    np.savetxt('result_rnn.csv', result, delimiter=',', header='Date,PredictedPrice,RealPrice', fmt='%1.3f', comments='')

    # show the training and saving is done
    print("Training done, data saved to result_rnn.csv")

main()
