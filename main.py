"""Точка входа в развитие нейронной сети"""
import logging
from optimizer import Optimizer
from tqdm import tqdm

# Настройка логов.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def train_networks(networks, dataset):
    """Обучим каждую сеть.

    Args:
        networks (list): Текущее количество сетей
        dataset (str): Набор данных для обучения / оценки
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Получим среднюю точность для группы сетей

    Args:
        networks (list): Список сетей

    Returns:
        float: Средняя точность популяции сетей.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices, dataset):
    """Создаем сеть с помощью генетического алгоритма.

    Args:
        generations (int): Сколько раз эволюционировать популяция
        population (int): Количество сетей в каждом поколении
        nn_param_choices (dict): Выбор параметров для сетей
        dataset (str): Набор данных для обучения / оценки

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Развиваем поколение.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Тренеруем и получаем точность для сетей
        train_networks(networks, dataset)

        # Получаем среднюю точность для этого поколения.
        average_accuracy = get_average_accuracy(networks)

        # Вывод средней точности каждого поколения.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Развиваем, кроме последней итерации.
        if i != generations - 1:
            # Эволюция
            networks = optimizer.evolve(networks)

    # Сортируем последнюю популяцию.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Выведем 5 лучших сетей    
    print_networks(networks[:5])

def print_networks(networks):
    """Выведем список сетей

    Args:
        networks (list): Население сетей
    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    """Развиваем сеть"""
    generations = 3  # Сколько раз будет эволиционировать популяция
    population = 5  # Количество сетей в каждом поколении
    dataset = 'iris'

    nn_param_choices = {
        'nb_neurons': [8, 16, 32],
        'nb_layers': [1, 2, 3],
        'activation': ['relu', 'elu', 'tanh'],
        'optimizer': ['rmsprop', 'adam', 'sgd'],
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices, dataset)

if __name__ == '__main__':
    main()
