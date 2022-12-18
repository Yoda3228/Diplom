"""
Класс, содержащий генетический алгоритм для развития.
"""
from functools import reduce
from operator import add
import random
from network import Network

class Optimizer():
    """Класс, реализующий оптимизацию"""

    def __init__(self, nn_param_choices, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):
        """Создаем оптимизатор.

        Args:
            nn_param_choices (dict): Возможные параметры сети
            retain (float): Процент популяции , который необходимо сохранить после каждого поколения
            random_select (float): Вероятность отклонения нейросети 
            mutate_chance (float): Вероятность, что нейросеть будет мутировать

        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        """Создаем случайню популяцию нейросети

        Args:
            count (int): размер популяции 

        Returns:
            (list): популяция сетевых объектов

        """
        pop = []
        for _ in range(0, count):
            # Создаем рандомную нейросеть
            network = Network(self.nn_param_choices)
            network.create_random()

            #Добавляем нейросеть к нашей популяции.
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """возвращает точность"""
        return network.accuracy

    def grade(self, pop):
        """Ищем средний показатель пригодности популяции.

        Args:
            pop (list): Население сети

        Returns:
            (float): средняя точность популяции

        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Делаем двух новый детей частью их потомков 

        Args:
            mother (dict): Параметры сети
            father (dict): Параметры сети

        Returns:
            (list): два сетевых объекта

        """
        children = []
        for _ in range(2):

            child = {}

            # Создаем параметры для ребенка.
            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Создаем сетевой объект.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Рандомно мутириуем несколько особей
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network):
        """Рандомно мутируем часть сети

        Args:
            network (dict): Параметры сети для изменения

        Returns:
            (Network): Случайно мутировавший сетевой объект

        """
        # Выбераем случайных ключ.
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # Заменяем один из параметров
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])

        return network

    def evolve(self, pop):
        """Развиваем популяцию сетей

        Args:
            pop (list): Список сетевых параметров

        Returns:
            (list): обученная популяция сетей

        """
        # Получаем оценку для каждой сети
        graded = [(self.fitness(network), network) for network in pop]

        # Сортировка по очкам.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Получаем номер, который мы хотим сохранить для следующего поколения
        retain_length = int(len(graded)*self.retain)

        # Родители - это вся сеть, которую мы хотим сохранить.
        parents = graded[:retain_length]

        # Часть отсеюващиейся  популяции оставляем
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Теперь узнаем, сколько мест нам осталось заполнить.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Добавим детей, выведенных из двух оставшихся сетей.
        while len(children) < desired_length:

            # Найдем случайных родителей
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Предположим, что это не одна и та же сеть
            if male != female:
                male = parents[male]
                female = parents[female]

                # Резведем их
                babies = self.breed(male, female)

                # добавляем детей по одному
        
                for baby in babies:
                    # Не увеличиваем длину больше желаемой
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
