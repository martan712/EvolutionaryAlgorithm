import random
import numpy as np
from weblogo import *
from Population import *

def generate_random_string(alphabet, length):
    return ''.join(random.choice(alphabet) for _ in range(length))
    
class GeneticAlgorithm:
    def __init__(self, alphabet, target, target_length, mu, N=200, K=5, iters=200):
        self.alphabet = alphabet
        self.target = target
        self.target_length = target_length
        self.mu = mu  # Mutation rate
        self.N = N  # Population size
        self.K = K  # Tournament selection size
        self.iters = iters  # Max iterations
        self.population = Population([generate_random_string(self.alphabet, self.target_length) for _ in range(N)])

    def fitness(self, individual):
        return sum(1 for t, i in zip(self.target, individual) if t == i) / self.target_length

    def tournament_selection(self):
        selected = random.sample(self.population.get_population(), min(self.K, len(self.population.get_population())))
        selected.sort(key=lambda ind: self.fitness(ind), reverse=True)
        return selected[0]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.target_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self,individual):
        mutated_individual = list(individual)
        for i in range(len(mutated_individual)):
            if np.random.choice((True, False), p=[self.mu, 1-self.mu]):
                mutated_individual[i] = random.choice(self.alphabet)  # Randomly mutate character
        return ''.join(mutated_individual)

    def create_new_population(self):
        new_population = []
        while len(new_population) < self.N:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            c1, c2 = self.crossover(p1, p2)
            c1, c2 = self.mutate(c1), self.mutate(c2)
            new_population.extend([c1, c2])
        self.population.set_population(new_population[:self.N])  # Keep population size fixed

    def visualize_png(self, inname, title, outname):
        fin = open(inname)
        sequences = read_seq_data(fin)
        logodata = LogoData.from_seqs(sequences)
        
        # Create the logo options
        logooptions = LogoOptions()
        logooptions.title = title
        
        # Increase the DPI (Resolution) for better quality
        logooptions.dpi = 300  # Default is usually 72 DPI, 300 is better quality
        
        # Optionally, adjust font sizes, scale, or other settings:
        logooptions.font_size = 40  # Adjust font size (default is usually around 12)
        logooptions.size = (24, 8)  # Adjust size of the logo, if necessary
        
        # Format the logo
        logoformat = LogoFormat(logodata, logooptions)
        png = png_formatter(logodata, logoformat)
        
        # Save the image at the high resolution
        with open(outname, 'wb') as file:
            file.write(png)
    
            
    def run(self):
        fitness_tracker = []
        fitnesses = [self.fitness(c) for c in self.population.get_population()]
        best_fitness = max(fitnesses)
        generation = 0
        while best_fitness != 1:
            if generation>= self.iters:
                break
            if generation%25 == 0:
                print(generation)
                print(best_fitness)
            generation+=1
            self.create_new_population()
            fitnesses = [self.fitness(c) for c in self.population.get_population()]
            best_fitness = max(fitnesses)
            fitness_tracker.append(best_fitness)
        print(f"Final Generation: {generation}")
        print(f"Best Fitness: {best_fitness:.4f}")
        
        filename=f"data/fasta/ex-mu{self.mu}-K{self.K}-N{self.N}-generation{generation}.fasta"
        self.population.generate_fasta(filename)
        self.visualize_png(filename, "title","output_image.png")
        print(self.population.min_hamming_distances())
        print(self.population.get_population())
        return fitness_tracker, [self.fitness(c) for c in self.population.get_population()]
