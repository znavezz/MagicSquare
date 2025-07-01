from magic_square_chromosome import MagicSquareChromosome
import numpy as np

class MostPerfectMagicSquareChromosome(MagicSquareChromosome):
    def __init__(self, n):
        super().__init__(n)
        if n % 4 != 0:
            raise ValueError("Most-perfect magic squares only exist for n divisible by 4.")
        self.s = n**2 + 1  # For 2x2 subsquares and diagonal pairs
    
    def clone(self):
        """
        Clones the current chromosome.
        """
        clone = MostPerfectMagicSquareChromosome(self.n)
        clone.square = np.copy(self.square)
        return clone

    def get_fitness(self):
        """
        Returns the fitness of the chromosome based on:
        - Standard magic square rules (rows/cols/diagonals = m)
        - 2x2 subsquares (wraparound) summing to 2s
        - Diagonal pairs (n/2 apart, wraparound) summing to s
        """
        fitness = super().get_fitness() + self.check_2x2_subsquares() + self.check_diagonal_pairs_n_2_apart()
        return fitness
    
    def check_2x2_subsquares(self):
        fitness = 0
        for i in range(self.n):
            for j in range(self.n):
                a = self.square[i][j]
                b = self.square[i][(j + 1) % self.n]
                c = self.square[(i + 1) % self.n][j]
                d = self.square[(i + 1) % self.n][(j + 1) % self.n]
                block_sum = a + b + c + d
                fitness += abs(block_sum - (2 * self.s))
        return fitness

    def check_diagonal_pairs_n_2_apart(self):
        half_n = self.n // 2
        fitness = 0
        for i in range(self.n):
            j = i
            j_opposite = (i + half_n) % self.n
            # Main diagonal pair
            val1 = self.square[i][i]
            val2 = self.square[j_opposite][j_opposite]
            fitness += abs((val1 + val2) - self.s)

            # Anti-diagonal pair
            val3 = self.square[i][self.n - 1 - i]
            val4 = self.square[j_opposite][self.n - 1 - j_opposite]
            fitness += abs((val3 + val4) - self.s)
        return fitness
    
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
        child = MostPerfectMagicSquareChromosome(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if np.random.rand() < 0.5:
                    child.square[i][j] = self.square[i][j]
                else:
                    child.square[i][j] = other.square[i][j]

        child.fix_missing_numbers()
        return child
