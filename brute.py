"""Обход каждой комбинации гиперпараметров"""
import logging
from network import Network
from tqdm import tqdm

# Настройка логов.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='brute-log.txt'
)

def train_networks(networks, dataset):
    """Обучаем сеть

    Args:
        networks (list): Текущее количество сетей
        dataset (str): Набор данных для обучения / оценки
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        network.print_network()
        pbar.update(1)
    pbar.close()

    # сортируем последнюю популяцию
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # выведем 5 лучших сетей
    print_networks(networks[:5])

def print_networks(networks):
    """распечатаем список сетей

    Args:
        networks (list): Население сетей

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def generate_network_list(nn_param_choices):
    """Сгенерирум список всех возможных сетей.

    Args:
        nn_param_choices (dict): Выбор параметров

    Returns:
        networks (list): Список сетевых объектов

    """
    networks = []

    
    for nbn in nn_param_choices['nb_neurons']:
        for nbl in nn_param_choices['nb_layers']:
            for a in nn_param_choices['activation']:
                for o in nn_param_choices['optimizer']:

                    # Установим параметры
                    network = {
                        'nb_neurons': nbn,
                        'nb_layers': nbl,
                        'activation': a,
                        'optimizer': o,
                    }

                    # Создаем экземпляр сетевого объекта с заданными параметрами
                    network_obj = Network()
                    network_obj.create_set(network)

                    networks.append(network_obj)

    return networks

def main():
    """тест брутфорсом  сети."""
    dataset = 'cifar10'

    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }

    logging.info("***Brute forcing networks***")

    networks = generate_network_list(nn_param_choices)

    train_networks(networks, dataset)

if __name__ == '__main__':
    main()
