import pandas as pd
from datetime import datetime as dt
from matplotlib import pyplot as plt
from sklearn.preprocessing import (LabelEncoder,
                                   MinMaxScaler)
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM

# ref: https://yq.aliyun.com/articles/174270


def parse(year, month, day, hour):
    return dt(year=int(year), month=int(month), day=int(day), hour=int(hour))


def clean_data():
    data = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv',
                       parse_dates={'date': ['year', 'month', 'day', 'hour']},
                       index_col=0,
                       date_parser=parse)

    data.drop('No', axis=1, inplace=True)
    data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    data['pollution'].fillna(0, inplace=True)
    data[24:].to_csv('pollution.csv')
    return


def load_data(plot=False):
    data = pd.read_csv('pollution.csv', header=0, index_col=0)

    if plot:
        plt.figure()
        groups = [0, 1, 2, 3, 5, 6, 7]
        i = 1
        for group in groups:
            plt.subplot(len(groups), 1, i)
            plt.plot(data[data.columns[group]].values)
            plt.title(data.columns[group], y=0.5, loc='right')
            i += 1

        plt.show()
    return data


def lstm_model(train_X, train_y, test_X, test_y, plot=False):
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # fit network
    fit = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y))

    # plot history
    if plot:
        plt.plot(fit.history['loss'], label='train')
        plt.plot(fit.history['val_loss'], label='test')
        plt.legend()
        plt.show()
    return


if __name__ == '__main__':
    data = load_data()
    encoder = LabelEncoder()
    data.iloc[:, 4] = encoder.fit_transform(data.iloc[:, 4])
    data = data.astype('float32')
    data['pred'] = data['pollution'].shift(-1)
    data.dropna(inplace=True)

    # train-test split
    n_train_hours = 365 * 24
    train, test = data.iloc[: n_train_hours, :].values, data.iloc[n_train_hours:, :].values

    # scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)
    test = scaler.fit_transform(test)

    # split features and labels
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    lstm_model(train_X, train_y, test_X, test_y, plot=True)