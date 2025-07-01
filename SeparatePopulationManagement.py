import numpy as np
from magic_square_chromosome import MagicSquareChromosome
from most_perfect_magic_square_chromosome import MostPerfectMagicSquareChromosome
import random

class IslandPopulationManagement:
    def __init__(self, N, size, state=0, most_state=False, mutation_rate=0.1, elitism=0.1):
        # Ensure population size is divisible by 4 for equal island sizes
        if size % 4 != 0:
            size = ((size // 4) + 1) * 4  # Round up to nearest multiple of 4
        
        self.island_size = size // 4
        self.total_size = size
        self.most_state = most_state
        
        # Create 4 islands
        self.islands = []
        for i in range(4):
            if self.most_state:
                island_pop = [MostPerfectMagicSquareChromosome(N) for _ in range(self.island_size)]
            else:
                island_pop = [MagicSquareChromosome(N) for _ in range(self.island_size)]
            self.islands.append(island_pop)
        
        self.generation = 0
        self.state = state
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.size = size
        self.migration_interval = 100
        self.migration_rate = 1/6  # 1/6 of population migrates
    
    # getters
    def get_population(self):
        """Returns flattened population from all islands"""
        population = []
        for island in self.islands:
            population.extend(island)
        return population
    
    def get_state(self):
        return self.state
    
    def get_generation(self):
        return self.generation
    
    def get_best_chromosomes(self, rate=0.1):
        """Get best chromosomes from entire population"""
        all_pop = self.get_population()
        sorted_pop = sorted(all_pop, key=lambda c: c.get_fitness())
        k = max(1, int(rate * len(sorted_pop)))
        return sorted_pop[:k]
    
    def get_best_chromosomes_from_island(self, island_idx, rate=0.1):
        """Get best chromosomes from a specific island"""
        island = self.islands[island_idx]
        sorted_pop = sorted(island, key=lambda c: c.get_fitness())
        k = max(1, int(rate * len(sorted_pop)))
        return sorted_pop[:k]
    
    def select_parent(self, island_idx):
        """Select parent from specific island using roulette wheel"""
        island = self.islands[island_idx]
        fitnesses = np.array([c.get_fitness() for c in island], dtype=float)
        max_fitness = np.max(fitnesses)
        fitnesses = max_fitness - fitnesses
        total = fitnesses.sum()
        
        if total <= 0:
            probs = np.ones(len(fitnesses)) / len(fitnesses)
        else:
            probs = fitnesses / total
            
        idx = np.random.choice(len(island), p=probs)
        return island[idx]
    
    def migrate_population(self):
        """Migrate 1/6 of population between islands every migration_interval generations"""
        if self.generation % self.migration_interval != 0:
            return
        
        migration_size = max(1, int(self.island_size * self.migration_rate))
        
        # Get migrants from each island (worst performers)
        migrants = []
        for i in range(4):
            island = self.islands[i]
            sorted_island = sorted(island, key=lambda c: c.get_fitness(), reverse=True)  # worst first
            island_migrants = sorted_island[:migration_size]
            migrants.extend(island_migrants)
            
            # Remove migrants from original island
            for migrant in island_migrants:
                island.remove(migrant)
        
        # Redistribute migrants randomly across islands
        random.shuffle(migrants)
        for i, migrant in enumerate(migrants):
            target_island = i % 4
            self.islands[target_island].append(migrant)
    
    def evaluate_population(self):
        """Evaluate each island separately, then handle migration"""
        # Process each island
        for island_idx in range(4):
            self._evaluate_island(island_idx)
        
        # Handle migration
        self.migrate_population()
        
        self.generation += 1
    
    def _evaluate_island(self, island_idx):
        """Evaluate a single island (same logic as original populationManagement)"""
        island = self.islands[island_idx]
        new_island = []
        
        if self.state != 2:
            elite = self.get_best_chromosomes_from_island(island_idx, self.elitism)
        else:
            elite = []

        new_island = []
        for chrom in elite:
            new_island.append(chrom)
        
        # regular/darwin/lamarck state
        if self.state != 0:
            copy_island = [chrom.clone() for chrom in island]
            for i in range(len(copy_island)):
                island[i].local_optimize()
            
            if self.state == 2:
                # elitism for lamarck
                lam_elite = self.get_best_chromosomes_from_island(island_idx, self.elitism)
                for chrom in lam_elite:
                    new_island.append(chrom)
        
        # crossover
        parents = []
        num_children = self.island_size - len(new_island)
        
        for i in range(num_children):
            parent1 = self.select_parent(island_idx)
            parent2 = self.select_parent(island_idx)
            parents.append((parent1, parent2))
        
        if self.state == 1:
            self.islands[island_idx] = copy_island
        
        for i in range(num_children):
            parent1, parent2 = parents[i]
            child = parent1.cross_over(parent2)
            new_island.append(child)
        
        # mutation
        for i in range(len(new_island)):
            if random.random() < self.mutation_rate:
                new_island[i].mutate()
        
        # update island
        self.islands[island_idx] = new_island
    
    def get_island_best_fitness(self, island_idx):
        """Get best fitness from a specific island"""
        island = self.islands[island_idx]
        return min(chrom.get_fitness() for chrom in island)
    
    def get_overall_best_fitness(self):
        """Get best fitness from entire population"""
        return min(self.get_island_best_fitness(i) for i in range(4))
    
    def print_island_stats(self):
        """Print statistics for each island"""
        print(f"Generation {self.generation}:")
        for i in range(4):
            best_fitness = self.get_island_best_fitness(i)
            avg_fitness = np.mean([chrom.get_fitness() for chrom in self.islands[i]])
            print(f"  Island {i}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}, Size={len(self.islands[i])}")
        print(f"  Overall Best: {self.get_overall_best_fitness():.2f}")