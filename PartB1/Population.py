from weblogo import *
from collections import Counter

from Utilities import *

class Population:
    
    def __init__(self, population):
        self.population = population
        self.L = len(self.population[0])

    def set_population(self, new_population):
        self.population=new_population
        
    def get_population(self):
        return self.population 
            
    def min_hamming_distance(self, x):
        dists = [hamming_distance(x, y) for y in self.population if x!=y]
        if len(dists) == 0:
            return 0
        return min(dists)
    def min_hamming_distances(self):
        return [self.min_hamming_distance(x) for x in self.population]
    
    def generate_fasta(self,filename):
        create_file_dir(filename)
        with open(filename, "w") as f:
            for i, seq in enumerate(self.population):
                f.write(f">sequence_{i}\n{seq}\n")
                
    def positionwise_entropy(self):
        alphabet = set(''.join(self.population))  # All unique characters in the population

        entropies = []

        for i in range(self.L):  # Iterate over each position in the strings
            position_counts = Counter(seq[i] for seq in self.population)
            position_values = [position_counts[char] for char in alphabet]  # Get the counts for each char at position i
            entropy = shannon_entropy(position_values)  # Compute the entropy at position i
            entropies.append(entropy)

        return entropies