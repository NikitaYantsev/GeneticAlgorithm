import numpy as np
import optuna
import tqdm.notebook


with open('genetic_input') as fin:
    task_number = int(fin.readline())
    task_categories = np.array(list(map(int, fin.readline().split())))
    estimate_time_to_solve_tasks = np.array(list(map(float, fin.readline().split())))
    developer_number = int(fin.readline())
    developer_coefficients = np.array([list(map(float, fin.readline().split())) for i in range(developer_number)])


class GeneticAlgorithm:
    """
    Genetic algorithm implementation class.
    """
    task_categories: np.array
    estimate_time_to_solve_tasks: np.array
    developer_coefficients: np.array
    population_size: int
    selection_size: int
    child_mutation_prob: float
    gen_mutation_prob: float
    selected: np.array
    children: np.array

    def __init__(self,
                 task_categories: np.array,
                 estimate_time_to_solve_tasks: np.array,
                 developer_coefficients: np.array,
                 population_size: int,
                 selection_size: int,
                 child_mutation_prob: float,
                 gen_mutation_prob: float) -> None:
        """
        A constructor for a genetic algorithm class instance.
        :param task_categories: np.array of task difficulty categories. (size=n_tasks)
        :param estimate_time_to_solve_tasks: np.array of time estimations for solving each task. (size=n_tasks)
        :param developer_coefficients: np.array of developer coefficients for each task difficulty respectively
        (size=n_developers, n_task_categories)
        :param population_size: size of initial population, also represents how many individuals are going to born after
        each selection phase.
        :param selection_size: size of individuals to be selected at each iteration.
        :param child_mutation_prob: probability for an individual to mutate.
        :param gen_mutation_prob: probability for a gen of a given individual to mutate.
        """
        self.task_categories = task_categories
        self.estimate_time_to_solve_tasks = estimate_time_to_solve_tasks
        self.developer_coefficients = developer_coefficients
        self.population_size = population_size
        self.selection_size = selection_size
        self.child_mutation_prob = child_mutation_prob
        self.gen_mutation_prob = gen_mutation_prob
        self.rng = np.random.default_rng()
        self.population = self.rng.integers(1,
                                            self.developer_coefficients.shape[0] + 1,
                                            size=(self.population_size,
                                                  self.estimate_time_to_solve_tasks.shape[0]))

    def selection(self) -> None:
        """
        Selection phase. Picks self.selection_size individuals with a highest fitness value.
        :return:
        """
        self.selected = self.population[np.argsort(self.fitness())[-self.selection_size:]]

    def fitness(self) -> np.array:
        """
        Function to compute fitness of a population.
        Fitness is computed as 100 / T_max, where T_max is the longest time interval spent to complete the tasks among
        all developers within an individual.
        :return:
        """
        fitness = np.zeros(self.population.shape[0], dtype=float)
        for developer in range(1, self.developer_coefficients.shape[0] + 1):
            developer_indexes = np.where(self.population == developer)
            task_indexes = developer_indexes[1]
            bins = developer_indexes[0]
            dev_time = self.estimate_time_to_solve_tasks[task_indexes] * self.developer_coefficients[
                developer - 1, self.task_categories[task_indexes] - 1]
            current_fitness = np.bincount(bins, weights=dev_time)
            fitness = np.maximum(current_fitness, fitness)
        fitness = 100 / fitness
        return fitness

    def crossover(self) -> None:
        """
        Crossover stage.
        Commented part of code implements 1-point crossover.
        Uncommented part of code implements uniform crossover.
        :return:
        """
        new_count = self.population_size - self.selection_size
        parent1 = self.rng.integers(0, self.selection_size, size=new_count)
        parent2 = (self.rng.integers(1, self.selection_size, size=new_count) + parent1) % self.selection_size
        # point = self.rng.integers(1, self.population.shape[1] - 1, size=new_count)
        self.children = np.where(
            # np.arange(self.population.shape[1])[None] <= point[..., None],
            self.rng.integers(2, size=self.population.shape[1]),
            self.selected[parent1],
            self.selected[parent2]
        )

    def mutation(self) -> None:
        """
        Mutation phase.
        Picks random amount of individuals (chance to be picked is self.child_mutation_prob) and mutates their genes
        (each gen has a self.gen_mutation_prob to get mutated).
        :return:
        """
        mut_childs_mask = self.rng.choice(2, p=(1 - self.child_mutation_prob,
                                                self.child_mutation_prob),
                                          size=len(self.children)) > 0
        mut_childs = self.rng.integers(1, 11, size=(mut_childs_mask.sum(), self.population.shape[1]))
        gen_childs_mask = self.rng.random(size=mut_childs.shape) <= self.gen_mutation_prob
        self.children[mut_childs_mask] = np.where(gen_childs_mask,
                                                  mut_childs,
                                                  self.children[mut_childs_mask])

    def step(self) -> None:
        """
        Computes one step of algorithm learning stage.
        This involves selection, crossover and mutation stages.
        After this, selected population and born population get combined.
        :return:
        """
        self.selection()
        self.crossover()
        self.mutation()
        self.population = np.concatenate([self.selected, self.children], axis=0)

    def fit(self, iterations: int) -> None:
        """
        Fits a genetic algorithm.
        :return:
        """
        pbar = tqdm.notebook.trange(iterations)
        for _ in pbar:
            self.step()
            pbar.set_description(f'{self.fitness().max()}')


def objective(trial):
    bound = trial.suggest_int("bound", 3, 99)
    population_size = trial.suggest_int("population_size", bound, 100)
    selection_size = trial.suggest_int("selection_size", 2, bound)
    child_mutation_prob = trial.suggest_float("child_mutation_prob", 0, 1)
    gen_mutation_prob = trial.suggest_float("gen_mutation_prob", 0, 1)

    ga = GeneticAlgorithm(task_categories,
                          estimate_time_to_solve_tasks,
                          developer_coefficients,
                          population_size,
                          selection_size,
                          child_mutation_prob,
                          gen_mutation_prob)
    ga.fit(1000)

    max_accuracy = ga.fitness().max()

    best_population = ga.population[np.argmax(ga.fitness())]
    with open('output.txt', 'a') as fout:
        print(f'{max_accuracy}:', file=fout)
        print(' '.join(list(map(str, best_population))), file=fout)
        print(file=fout)
        print(file=fout)

    return max_accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
