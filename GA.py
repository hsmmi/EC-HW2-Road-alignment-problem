from matplotlib import pyplot as plt
import numpy as np
from map import map


class GA:
    def __init__(
        self,
        pop_size,
        chromosome_length,
        chromosome_value,
        max_iter,
        pc,
        pm,
        fitness_func,
    ):
        self.pop_size = pop_size
        self.chromosome_length = chromosome_length
        self.chromosome_value = chromosome_value
        self.max_iter = max_iter
        self.pc = pc
        self.pm = pm
        self.fitness_func = fitness_func
        self.population = []
        self.fitness = []
        self.best_fitness = []
        self.best_chromosome = []
        self.avg_fitness = []
        self.init_population()
        self.run()

    def init_population(self):
        for i in range(self.pop_size):
            chromosome = np.random.randint(
                0, self.chromosome_value, self.chromosome_length
            ).tolist()
            self.population.append(chromosome)
            self.fitness.append(self.fitness_func(chromosome))
        self.best_fitness.append(min(self.fitness))
        self.best_chromosome.append(
            self.population[self.fitness.index(min(self.fitness))]
        )

    def selection(self):
        # Inverse fitness for maximize
        max_fitness = max(self.fitness) + 1
        inverse_fitness = [max_fitness - f for f in self.fitness]
        # Roulette wheel selection
        fitness_sum = sum(inverse_fitness)
        fitness_prob = [f / fitness_sum for f in inverse_fitness]
        fitness_prob_sum = np.cumsum(fitness_prob)

        new_population = []
        for i in range(self.pop_size):
            rand = np.random.rand()
            for j in range(self.pop_size):
                if rand < fitness_prob_sum[j]:
                    new_population.append(self.population[j])
                    break
        self.population = new_population

    def crossover(self):
        new_population = []
        new_fitness = []
        for i in range(0, self.pop_size, 2):
            rand = np.random.rand()
            if rand < self.pc:
                rand = np.random.randint(0, self.chromosome_length)
                offspring1 = (
                    np.hstack(
                        (self.population[i][:rand], self.population[i + 1][rand:])
                    )
                    .astype(int)
                    .tolist()
                )
                offspring2 = (
                    np.hstack(
                        (self.population[i + 1][:rand], self.population[i][rand:])
                    )
                    .astype(int)
                    .tolist()
                )
                new_population.append(offspring1)
                new_fitness.append(self.fitness_func(offspring1))
                new_population.append(offspring2)
                new_fitness.append(self.fitness_func(offspring2))
            else:
                new_population.append(self.population[i])
                new_fitness.append(self.fitness[i])
                new_population.append(self.population[i + 1])
                new_fitness.append(self.fitness[i + 1])

        self.population = new_population
        self.fitness = new_fitness

    def crossover2(self):
        new_population = []
        new_fitness = []
        while len(new_population) < self.pop_size:
            rand1 = np.random.randint(0, self.pop_size)
            rand2 = np.random.randint(0, self.pop_size)
            rand = np.random.rand()
            if rand < self.pc:
                # Choose crossover point as the crossover of two path
                # Find which gene is the same
                same_gene = []
                for j in range(self.chromosome_length):
                    if self.population[rand1][j] == self.population[rand2][j]:
                        same_gene.append(j)
                # Choose crossover point
                if len(same_gene) == 0:
                    rand = np.random.randint(0, self.chromosome_length)
                else:
                    rand = np.random.choice(same_gene)

                offspring1 = (
                    np.hstack(
                        (self.population[rand1][:rand], self.population[rand2][rand:])
                    )
                    .astype(int)
                    .tolist()
                )
                offspring2 = (
                    np.hstack(
                        (self.population[rand2][:rand], self.population[rand1][rand:])
                    )
                    .astype(int)
                    .tolist()
                )
                # Check if offspring is not in new population
                if offspring1 not in new_population:
                    new_population.append(offspring1)
                    new_fitness.append(self.fitness_func(offspring1))
                if offspring2 not in new_population:
                    new_population.append(offspring2)
                    new_fitness.append(self.fitness_func(offspring2))
        self.population = new_population[: self.pop_size]
        self.fitness = new_fitness[: self.pop_size]

    def crossover3(self):
        new_population = []
        new_fitness = []
        for i in range(0, self.pop_size, 2):
            rand = np.random.rand()
            if rand < self.pc:
                # Choose crossover point as the crossover of two path
                # Find which gene is the same
                same_gene = []
                for j in range(self.chromosome_length):
                    if self.population[i][j] == self.population[i + 1][j]:
                        same_gene.append(j)
                # Choose crossover point
                if len(same_gene) == 0:
                    rand = np.random.randint(0, self.chromosome_length)
                else:
                    rand = np.random.choice(same_gene)

                offspring1 = (
                    np.hstack(
                        (self.population[i][:rand], self.population[i + 1][rand:])
                    )
                    .astype(int)
                    .tolist()
                )
                offspring2 = (
                    np.hstack(
                        (self.population[i + 1][:rand], self.population[i][rand:])
                    )
                    .astype(int)
                    .tolist()
                )
                new_population.append(offspring1)
                new_fitness.append(self.fitness_func(offspring1))
                new_population.append(offspring2)
                new_fitness.append(self.fitness_func(offspring2))
            else:
                new_population.append(self.population[i])
                new_fitness.append(self.fitness[i])
                new_population.append(self.population[i + 1])
                new_fitness.append(self.fitness[i + 1])
        self.population = new_population[: self.pop_size]
        self.fitness = new_fitness[: self.pop_size]

    def mutation(self):
        for i in range(self.pop_size):
            rand = np.random.rand()
            if rand < self.pm:
                rand = np.random.randint(0, self.chromosome_length)
                self.population[i][rand] = np.random.randint(0, self.chromosome_value)
                self.fitness[i] = self.fitness_func(self.population[i])

    def mutation2(self):
        for i in range(self.pop_size):
            rand = np.random.rand()
            if rand < self.pm:
                rand = np.random.randint(0, self.chromosome_length)
                if self.population[i][rand] == 0:
                    self.population[i][rand] = 1
                elif self.population[i][rand] == self.chromosome_value - 1:
                    self.population[i][rand] = self.chromosome_value - 2
                else:
                    self.population[i][rand] += np.random.choice([-1, 1])

    def run(self):
        for i in range(self.max_iter):
            self.selection()
            self.crossover()
            # self.crossover2()
            # self.crossover3()
            self.mutation()
            # self.mutation2()

            # Add best chromosome to new population
            # If not exist in new population
            if self.best_chromosome[-1] not in self.population:
                self.population[-1] = self.best_chromosome[-1]
                self.fitness[-1] = self.best_fitness[-1]

            self.best_fitness.append(min(self.fitness))
            self.best_chromosome.append(
                self.population[self.fitness.index(min(self.fitness))]
            )
            self.avg_fitness.append(sum(self.fitness) / len(self.fitness))
        return self.best_fitness, self.best_chromosome


mp = map("my_map.npy")


def plot_fitness():
    # Plot avg and best fitness for 10 run and their avg
    max_iter = 500
    avg_best_fitness = np.zeros(max_iter + 1)
    avg_avg_fitness = np.zeros(max_iter)
    # 2 plot in a row
    fig, axs = plt.subplots(1, 2)
    # Fig size
    fig.set_size_inches(18, 7)
    for i in range(10):
        ga = GA(
            pop_size=100,
            chromosome_length=mp.width,
            chromosome_value=mp.height,
            max_iter=max_iter,
            pc=0.8,
            pm=0.2,
            fitness_func=mp.calc_dist,
        )
        axs[0].plot(ga.best_fitness, label="Run " + str(i + 1), alpha=0.25)
        axs[1].plot(ga.avg_fitness, label="Run " + str(i + 1), alpha=0.25)
        avg_best_fitness += ga.best_fitness
        avg_avg_fitness += ga.avg_fitness
    avg_best_fitness /= 10
    avg_avg_fitness /= 10
    axs[0].plot(avg_best_fitness, label="Avg")
    axs[1].plot(avg_avg_fitness, label="Avg")
    axs[0].set_title("Best fitness setup Traditional GA")
    axs[1].set_title("Average fitness setup Traditional GA")
    axs[0].legend()
    axs[1].legend()
    plt.show()


plot_fitness()

# ga = GA(
#     pop_size=100,
#     chromosome_length=mp.width,
#     chromosome_value=mp.height,
#     max_iter=10000,
#     pc=0.8,
#     pm=0.2,
#     fitness_func=mp.calc_dist,
# )

# print(ga.best_fitness[-1])
# print(ga.best_chromosome[-1])
# mp.print_path(ga.best_chromosome[-1])

print("pause")
