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
        max_fitness = max(self.fitness)
        inverse_fitness = [2 * max_fitness - f for f in self.fitness]
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
        # Sort population by fitness
        population_sorted = [
            x
            for _, x in sorted(
                zip(self.fitness, self.population), key=lambda pair: pair[0]
            )
        ]
        new_population = population_sorted[:4]
        # new_population = population_sorted[:4]

        # Add two best chromosome to new population

        while len(new_population) < self.pop_size:
            rand = np.random.rand()
            if rand < self.pc:
                rand1 = np.random.randint(0, self.pop_size)
                rand2 = np.random.randint(0, self.pop_size)
                rand = np.random.randint(0, self.chromosome_length)
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
                if offspring2 not in new_population:
                    new_population.append(offspring2)
        self.population = new_population[: self.pop_size]

    def mutation(self):
        for i in range(self.pop_size):
            rand = np.random.rand()
            if rand < self.pm:
                rand = np.random.randint(0, self.chromosome_length)
                self.population[i][rand] = np.random.randint(0, self.chromosome_value)
                self.fitness[i] = self.fitness_func(self.population[i])

    def run(self):
        for i in range(self.max_iter):
            self.selection()
            self.crossover()
            self.mutation()
            # Add best chromosome to new population
            self.population[-1] = self.best_chromosome[-1]
            self.fitness[-1] = self.best_fitness[-1]

            self.best_fitness.append(min(self.fitness))
            self.best_chromosome.append(
                self.population[self.fitness.index(min(self.fitness))]
            )
        return self.best_fitness, self.best_chromosome


mp = map("map.npy")

ga = GA(
    pop_size=100,
    chromosome_length=mp.width,
    chromosome_value=mp.height,
    max_iter=10000,
    pc=1,
    pm=0.9,
    fitness_func=mp.calc_dist,
)

print(ga.best_fitness[-1])
print(ga.best_chromosome[-1])
mp.print_path(ga.best_chromosome[-1])

print("pause")
