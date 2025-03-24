import random
import numpy as np
from fractions import Fraction
from collections import Counter

from Population import *
from Utilities import *

def generate_random_string(alphabet, length):
    return ''.join(random.choice(alphabet) for _ in range(length))
    
class GeneticAlgorithm:
    def __init__(self, alphabet, target, target_length, mu, N=200, K=5, iters=200, experiment_name=""):
        self.alphabet = alphabet
        self.target = target
        self.target_length = target_length
        self.mu = mu  # Mutation rate
        self.N = N  # Population size
        self.K = K  # Tournament selection size
        self.iters = iters  # Max iterations
        self.population = Population([generate_random_string(self.alphabet, self.target_length) for _ in range(self.N)])
        self.experiment_name = experiment_name
        self.experiment_id = 0
        
    def reset_population(self):
        self.population = Population([generate_random_string(self.alphabet, self.target_length) for _ in range(self.N)])
        
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
         
    def run(self):
        fitness_tracker = []
        fitnesses = [self.fitness(c) for c in self.population.get_population()]
        best_fitness = max(fitnesses)
        generation = 0
        while best_fitness != 1:
            if generation>= self.iters:
                break
            # if generation%25 == 0:
            #     print(generation)
            #     print(best_fitness)
            generation+=1
            self.create_new_population()
            fitnesses = [self.fitness(c) for c in self.population.get_population()]
            best_fitness = max(fitnesses)
            fitness_tracker.append(best_fitness)
        print(f"Final Generation: {generation}")
        print(f"Best Fitness: {best_fitness:.4f}")
        
        filename=f"ex-{self.experiment_name}-mu{ str(Fraction(self.mu).limit_denominator()).replace('/', 'div')}-K{self.K}-N{self.N}.{self.experiment_id}"
        self.population.generate_fasta("data/fasta/"+filename+".fasta")
        visualize_png("data/fasta/"+filename+".fasta", "title","figures/logo/"+filename+".png")
        dist_to_csv(Counter(self.population.min_hamming_distances()), "data/csv/"+filename+".csv")
        
        self.experiment_id+=1
        
        return fitness_tracker, [self.fitness(c) for c in self.population.get_population()]
