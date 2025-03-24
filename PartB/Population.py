from weblogo import *

def hamming_distance(s1, s2):
    return sum(x != y for x, y in zip(s1, s2))
   
class Population:
    
    def __init__(self, population):
        self.population = population

    def set_population(self, new_population):
        self.population=new_population
        
    def get_population(self):
        return self.population 
            
    def min_hamming_distances(self):
        return [min([hamming_distance(x, y) for y in self.population if x!=y]) for x in self.population]
        
    def generate_fasta(self,filename):
        with open(filename, "w") as f:
            for i, seq in enumerate(self.population):
                f.write(f">sequence_{i}\n{seq}\n")