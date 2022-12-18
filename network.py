"""Класс, отвечающий за развитие нейросети """
import random
import logging
from train import train_and_score

class Network():
    """Пока что ограничен MLPS """

    def __init__(self, nn_param_choices=None):
        """Инициализация нейросети

        Args:
            nn_param_choices (dict): Параметры для нейросети, библиотеки:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): представляет параметры MLPS

    def create_random(self):
        """Создаем случайную нейросеть"""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Устанавливаем свойства сети

        Args:
            network (dict): параметры сети

        """
        self.network = network

    def train(self, dataset):
        """Обучаем нейросеть и фиксируем точность

        Args:
            dataset (str): название датасета

        """
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network, dataset)

    def print_network(self):
        """вывод сети"""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
