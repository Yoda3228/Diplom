"""
Утилита, используемая классом Network для фактического обучения.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from pandas import DataFrame
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

# Ранняя остановка
early_stopper = EarlyStopping(patience=5)

def get_iris():
# Устанавливаем значения по умолчанию
    nb_classes = 10
    batch_size = 128
    input_shape = (250,)

# Получаем данные
    iris = datasets.load_iris()
    iris_frame = DataFrame(iris.data)
    iris_frame.columns = iris.feature_names
    iris_frame['target'] = iris.target
    iris_frame['name'] = iris_frame.target.apply(lambda x : iris.target_names[x])
    iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']].corr()
    corr = iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']].corr()
    x_train, y_train, x_test, y_test = train_test_split(iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']], iris_frame['target'], test_size = 0.5, random_state = 0)
# Преобразовываем векторы классов в матрицы двоичных классов
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test,y_train, y_test)


def get_cifar10():
    """Получаем набор данных CIFAR и обработываем данные"""
    # Устанавливаем значения по умолчанию
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Получаем данные
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Преобразовываем векторы классов в матрицы двоичных классов
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist():
    """Получаем набор данных MNIST и обработываем данные"""
    # Устанавливаем значения по умолчанию
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Получаем данные
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Преобразовываем векторы классов в матрицы двоичных классов
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def compile_model(network, nb_classes, input_shape):
    """Составляем последовательную модель

    Args:
        network (dict): параметры сети

    Returns:
        скомпилированная сеть

    """
    # Получаем наши сетевые параметры
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Добавим слой
    for i in range(nb_layers):

        # Нужна входная форма для первого слоя.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # запрограммированный отсев

    # Выходной слой.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset):
    """Обучаем модель, вернем тестовую потерю

    Args:
        network (dict): параметры сети
        dataset (str): Набор данных для обучения / оценки

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist()
    elif dataset == 'iris':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_iris()

    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # с использованием ранней остановки, поэтому нет реальных ограничений
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 точность 0 потери.
