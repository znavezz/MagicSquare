import numpy as np
from magic_square_chromosome import MagicSquareChromosome
from most_perfect_magic_square_chromosome import MostPerfectMagicSquareChromosome
import random

class populationManagement:
    def __init__(self, N, size ,state = 0, most_state = False, mutation_rate = 0.1, elitism = 0.1):
        if most_state:
            self.population = [MostPerfectMagicSquareChromosome(N) for i in range(size)]
        else:
            self.population = [MagicSquareChromosome(N) for i in range(size)]
        self.generation = 0
        self.state = state
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.size = size

    # getters
    def get_population(self):
        return self.population
    
    
    def get_state(self):
        return self.state
    
    def get_generation(self):
        return self.generation
    
    def get_best_chromosomes(self, rate = 0.1):
        sorted_pop = sorted(self.population,key=lambda c: c.get_fitness())
        k = max(1, int(rate * len(sorted_pop)))
        return sorted_pop[:k]
    
    def select_parent(self):
        # roulette wheel
        fitnesses = np.array([c.get_fitness() for c in self.population], dtype=float)
        max_fitness = np.max(fitnesses)
        fitnesses = max_fitness - fitnesses
        total = fitnesses.sum()
        if total <= 0:
            probs = np.ones(len(fitnesses)) / len(fitnesses)
        else:
            probs = fitnesses / total
        idx = self.rng.choice(len(self.population), p=probs) \
              if hasattr(self, 'rng') else \
              np.random.choice(len(self.population), p=probs)
        return self.population[idx]
    
    def evaluate_population(self):
        new_population = []

        if self.state != 2:
            # elitism
            alit = self.get_best_chromosomes(self.elitism)
            for i in alit:
                new_population.append(i)

        # regular/darwin/lamarck atate
        if self.state != 0:
            copy_population = [chrom.clone() for chrom in self.population]
            for i in range(len(copy_population)):
                self.population[i].local_optimize()
        if self.state == 2:
            # elitism
            alit = self.get_best_chromosomes(self.elitism)
            for i in alit:
                new_population.append(i)

        # crossover
        parets = []
        num_children = int(self.size * (1 - self.elitism))
        for i in range(num_children):
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            parets.append((parent1, parent2))

        if self.state == 1:
            self.population = copy_population

        for i in range(num_children):
            parent1, parent2 = parets[i]
            child = parent1.cross_over(parent2)
            new_population.append(child)
        
        

        # mutation
        for i in range(len(new_population)):
            if random.random() < self.mutation_rate:
                new_population[i].mutate()

        # update population
        self.population = new_population
        self.generation += 1