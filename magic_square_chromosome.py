import copy
import numpy as np
from random import sample
from math import sqrt
class MagicSquareChromosome:
    def __init__(self, n):
        self.n = n
        self.square = self.get_random_chromosome()
        self.age = 0
        self.m = n * (n**2 + 1) // 2

    def get_random_chromosome(self):
        """
        Generates a random chromosome for the magic square.
        """
        numbers = list(range(1, self.n**2 + 1))
        random_numbers = sample(numbers, len(numbers))
        square = np.array(random_numbers).reshape(self.n, self.n)
        return square

    def get_flat(self):
        """
        Flattens the 2D square into a 1D list.
        """
        flat = []
        for row in self.square:
            flat.extend(row)
        return flat
    def get_square(self):
        """
        Returns the 2D square.
        """
        return self.square
    def get_age(self):
        """
        Returns the age of the chromosome.
        """
        return self.age
    def increment_age(self):
        """
        Increments the age of the chromosome.
        """
        self.age += 1
    def get_n(self):
        """
        Returns the size of the square.
        """
        return self.n
    def get_fitness(self, fitness = 0):
        """
        Returns the fitness of the chromosome.
        """
        # Check rows
        for row in self.square:
            fitness += np.abs(sum(row) - self.m)
        # Check columns
        for col in range(self.n):
            fitness += np.abs(sum(self.square[:, col]) - self.m)
        # Check diagonals
        fitness += np.abs(sum(self.square[i][i] for i in range(self.n)) - self.m)
        fitness += np.abs(sum(self.square[i][self.n - 1 - i] for i in range(self.n)) - self.m)
        return fitness
    
    def clone(self):
        """
        Clones the current chromosome.
        """
        clone = MagicSquareChromosome(self.n)
        clone.square = np.copy(self.square)
        return clone
    def mutate(self):
        """
        Perform k random pair-swaps where  

            k = max(1, n // 2)

        so for n = 3 you still mutate once, for n = 10 you mutate 5 times,
        etc.  You keep the nice “swap two cells” semantics, just more of
        them when the board is larger.
        """
        k = max(1, self.n // 2)
        for _ in range(k):
            (i, j), (k_, l) = np.random.randint(0, self.n, size=(2, 2))
            self.square[i, j], self.square[k_, l] = self.square[k_, l], self.square[i, j]


    def cross_over(self, other):
        """
        Performs crossover between this chromosome and another chromosome.
        
        The crossover operation creates a new chromosome by selecting each element
        from either this chromosome or the other chromosome with equal probability.
        
        Args:
            other: Another MagicSquareChromosome to cross over with
            
        Returns:
            A new MagicSquareChromosome resulting from the crossover
        """
        child = MagicSquareChromosome(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if np.random.rand() < 0.5:
                    child.square[i][j] = self.square[i][j]
                else:
                    child.square[i][j] = other.square[i][j]

        child.fix_missing_numbers()
        return child
    
    def local_optimize(self):
        """
        Try up to  n  improving swaps between “heavy” rows/cols and
        “light” rows/cols.  The loop breaks early as soon as a swap really
        improves the fitness (greedy first-improvement local search).
        """
        attempts = int(np.ceil(self.n))          # 1, 2, 3, …
        for _ in range(attempts):
            original_fitness = self.get_fitness()

            # --- select heavy / light rows & cols once per attempt ----
            row_sums = self.square.sum(axis=1)
            col_sums = self.square.sum(axis=0)

            max_row_idx = row_sums.argmax()
            min_row_idx = row_sums.argmin()
            max_col_idx = col_sums.argmax()
            min_col_idx = col_sums.argmin()

            improved = False

            # -- row swap ------------------------------------------------
            if row_sums[max_row_idx] > self.m and row_sums[min_row_idx] < self.m:
                j, k = np.random.randint(0, self.n, size=2)
                self.square[max_row_idx, j], self.square[min_row_idx, k] = \
                    self.square[min_row_idx, k], self.square[max_row_idx, j]
                improved = self.get_fitness() < original_fitness
                if not improved:      # undo if it hurt
                    self.square[max_row_idx, j], self.square[min_row_idx, k] = \
                        self.square[min_row_idx, k], self.square[max_row_idx, j]

            # -- column swap --------------------------------------------
            if not improved and col_sums[max_col_idx] > self.m and col_sums[min_col_idx] < self.m:
                i, k = np.random.randint(0, self.n, size=2)
                self.square[i, max_col_idx], self.square[k, min_col_idx] = \
                    self.square[k, min_col_idx], self.square[i, max_col_idx]
                improved = self.get_fitness() < original_fitness
                if not improved:      # undo if it hurt
                    self.square[i, max_col_idx], self.square[k, min_col_idx] = \
                        self.square[k, min_col_idx], self.square[i, max_col_idx]

            if improved:      # greedy first-improvement
                return 1

        return 0
    
    def fix_missing_numbers(self):
        """
        Fixes the missing numbers in the magic square.
        
        This method ensures that all numbers from 1 to n^2 are present in the square by replacing the ones that appear more than once with thouse who dont appear at all.
        """
        numbers = list(range(1, self.n**2 + 1))
        flat = self.get_flat()

        seen = set()
        duplicates = []
        for num in flat:
            if num in seen:
                duplicates.append(num)
            else:
                seen.add(num)

        missing = list(set(numbers) - seen)
        missing.sort()

        idx = 0
        used = set()
        for i in range(self.n):
            for j in range(self.n):
                val = self.square[i][j]
                # If we've seen this number before and it's a duplicate
                if val in used:
                    self.square[i][j] = missing.pop(0)
                else:
                    used.add(val)





    