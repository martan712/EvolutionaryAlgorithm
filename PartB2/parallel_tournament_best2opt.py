import csv
import math
import functools
import random
import matplotlib.pyplot as plt
import itertools
from operator import itemgetter
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run a TSP optimization experiment.")
    parser.add_argument('--filename', type=str, required=True, help="Path to the TSP file (e.g., kroA150.tsp)")
    parser.add_argument('--file_ext', type=str, required=True, help="File extension (e.g., kroA150)")
    parser.add_argument('--timeout', type=int, required=True, help="File extension (e.g., kroA150)")
    return parser.parse_args()

# Command-line arguments
args = parse_args()
filename = args.filename
file_ext = args.file_ext
timeout= int(args.timeout)


distance_matrix = []

def parallel_best_2_opt(route, max_depth):
    route.best_2_opt_local_search(max_depth)
    return route

class City:
    ID: int
    x: float
    y: float

    def __init__(self, x: float, y: float, id: int):
        self.x = x
        self.y = y
        self.ID = id

    def dist(self, other):
        return math.dist((self.x, self.y), (other.x, other.y))

    def __str__(self):
        return f'{self.x}-{self.y}'
    __repr__ = __str__
    
class Route:
    cities: [City]

    def __init__(self, cities: [City]):
        self.cities = cities

    def distance(self):
        total = 0.0
        n = len(self.cities)
        for i in range(n):
            total += distance_matrix[self.cities[i].ID][self.cities[(i + 1) % n].ID]  # Ensures loop closure
        return total
    
    def fitness(self):
        return 1/self.distance()

    def crossover(self, other):
        start = random.randint(0, len(self.cities))
        end = start
        
        while end == start:
            end = random.randint(0, len(self.cities))
        if start > end:
            x = end
            end = start
            start = x
            
        middle1 = self.cities[start:end]
        middle2 = other.cities[start:end]

        sides1 = [city for city in other.cities if not city in middle1]
        sides2 = [city for city in self.cities if not city in middle2]
        
        child1 = middle1 + sides1
        child2 = middle2 + sides2
        child1 = child1[len(child1)-start:]+child1[:len(child1)-start]
        child2 = child2[len(child1)-start:]+child2[:len(child1)-start]
        
        return (Route(child1), Route(child2))

    def mutate(self):
        a = random.randint(0, len(self.cities)-1)
        b = a
        while a == b:
            b = random.randint(0, len(self.cities)-1)
        
        tmp = self.cities[a]
        self.cities[a] = self.cities[b]
        self.cities[b] = tmp

    def two_opt_swap(self, i, j):
        self.cities[i:j + 1] = reversed(self.cities[i:j + 1])

    def best_2_opt_local_search(self, max_its=5):
        function_evals = 0
        n = len(self.cities)

        for _ in range(max_its):
            improved = False

            for i in range(n - 2):  # from Ci to Ci+1
                for j in range(i + 2, n - 1):  # from Cj to Cj+1
                    A, B = self.cities[i], self.cities[i + 1]
                    C, D = self.cities[j], self.cities[j + 1]

                    before = distance_matrix[A.ID][B.ID] + distance_matrix[C.ID][D.ID]
                    after = distance_matrix[A.ID][C.ID] + distance_matrix[B.ID][D.ID]
                    delta = before - after
                    function_evals += 1

                    if delta > 0:
                        # Perform Best-2-Opt style swap: reverse the segment between B and C (exclusive),
                        # which is just endpoints swapped in effect.
                        self.cities[i + 1:j + 1] = reversed(self.cities[i + 1:j + 1])
                        improved = True
                        break

                if improved:
                    break
            if not improved:
                break

        return function_evals
         
    def __str__(self):
        return "->".join(map(lambda city: str(city.ID), self.cities))
    __repr__ = __str__


class Population:
    def __init__(self, cities: [City], n_population: int, K: int):
        self.routes = []
        self.n_population = n_population
        self.K = K
        cities = cities.copy()
        for i in range(n_population):
            random.shuffle(cities)
            self.routes.append(Route(cities.copy()))
   
    def tournament_selection(self):
        selected = random.sample(self.routes, self.K)
        selected.sort(key=lambda x: x.fitness(), reverse=True)
        return selected[0]
    
    def create_new_population(self):
        new_population = []
        while len(new_population) < self.n_population:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            c1, c2 = p1.crossover(p2)
            new_population.extend([c1, c2])
        self.routes = new_population
        
    def local_search(self, max_depth: float):
        with ThreadPoolExecutor() as executor:
                self.routes = list(executor.map(parallel_best_2_opt, self.routes, [max_depth] * len(self.routes)))

    def mutate(self, mutation_rate: float):
        for route in self.routes:
            if random.uniform(0, 1) < mutation_rate:
                route.mutate()

    def plot_route(self, title="Route", filename=""):
        routes = [(route.fitness(), route) for route in self.routes]
        routes.sort(key=itemgetter(0), reverse=True)
        best_fit, best = routes[0]
        dist = best.distance()
        best = best.cities + [best.cities[0]]
        x = [city.x for city in best]
        y = [city.y for city in best]

        plt.figure(figsize=(10, 8), facecolor='white')
        plt.plot(x, y, 'b-', linewidth=1.5, alpha=0.8)
        plt.plot(x, y, 'o', markersize=8, 
                markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.5)

        for i, city in enumerate(best):
            plt.text(city.x, city.y, str(city.ID), 
                    fontsize=9, ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.7, 
                            edgecolor='none', pad=2))

        plt.gca().set_facecolor('white')
        plt.title(f"{file_ext} Solution - Total Distance: {dist:.2f}", 
                fontsize=12, pad=20)
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def print_best(self):
        routes = [(route.fitness(), route) for route in self.routes]
        routes.sort(key=itemgetter(0), reverse=True)
        routes = [route for fitness, route in routes]
        print(f"{routes[0].distance()}: {routes[0]}")

    def get_score(self) -> float:
        routes = [(route.fitness(), route) for route in self.routes]
        routes.sort(key=itemgetter(0), reverse=True)
        routes = [route for fitness, route in routes]
        return routes[0].fitness()

def run_experiment(params):
    n_population, K, mu, max_depth, timeout, cities = params
    population = Population(cities, n_population, K)
    t0 = perf_counter()
    t, score = [], []
    
    while perf_counter() - t0 < timeout:
        population.create_new_population()
        population.mutate(mu)
        population.local_search(max_depth)
        
        t.append(perf_counter() - t0)
        score.append(population.get_score())
    
    best_route = max(population.routes, key=lambda r: r.fitness())
    best_distance = best_route.distance()
    
    filename = f"results/{file_ext}_npop={n_population}_K={K}_mu={mu}_depth={max_depth}.png"
    population.plot_route(filename=filename)
    
    return {
        'parameters': {
            'n_population': n_population,
            'K': K,
            'mu': mu,
            'max_depth': max_depth
        },
        'best_route':str(best_route),
        'best_distance': best_distance,
        'time_series': list(zip(t, score)),
        'convergence_time': t[-1] if t else timeout
    }

def grid_search(cities, param_grid, timeout_per_run):
    all_combinations = list(itertools.product(
        param_grid['n_population'],
        param_grid['K'],
        param_grid['mu'],
        param_grid['max_depth']
    ))
    
    results = []
    tasks = [(comb[0], comb[1], comb[2], comb[3], timeout_per_run, cities.copy()) 
             for comb in all_combinations]  # Note: cities.copy() for thread safety
    
    for task in tasks:
        result = run_experiment(task)
        print(f"Completed task with result: {result['best_distance']:.2f}")
        results.append(result)
    return results

def analyze_and_plot_results(results, cities):
    if not results:
        print("No results to analyze!")
        return
    
    best_result = min(results, key=lambda r: r['best_distance'])
    print("\nBest parameter set:")
    for param, value in best_result['parameters'].items():
        print(f"{param}: {value}")
    print(f"Best distance: {best_result['best_distance']:.2f}")
    
    plt.figure(figsize=(12, 8))
    for result in results:
        if (result['parameters']['n_population'] == best_result['parameters']['n_population'] and
            result['parameters']['K'] == best_result['parameters']['K'] and
            result['parameters']['mu'] == best_result['parameters']['mu']):
            
            t, score = zip(*result['time_series'])
            label = f"depth={result['parameters']['max_depth']}"
            plt.plot(t, score, label=label)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Fitness (1/distance)")
    plt.title("Convergence for different local search depths")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/depth_comparison.png")
    plt.close()

# (Keep City, Route, and Population classes as before)

def swap_dict_values(d, index1, index2):
    keys = list(d)
    d[keys[index1]], d[keys[index2]] = d[keys[index2]], d[keys[index1]]
    
def run_experiment_for_depths(params):
    results = []
    experiment_params = params.copy()
    experiment_params['max_depth'] = experiment_params.pop('depths')
    
    for max_depth in params['depths']:
        print(f"Running experiment with max_depth={max_depth}...")
        
        # Package parameters for run_experiment
        experiment_params['max_depth'] = max_depth
        # Run experiment and collect result
        result = run_experiment([experiment_params["n_population"], 
                                 experiment_params["K"], 
                                 experiment_params["mu"], 
                                 experiment_params["max_depth"], 
                                 experiment_params["timeout"], 
                                 experiment_params["cities"]])
        results.append(result)

    return results

def plot_combined_distance_vs_time(results, n_population, K, mu):
    plt.figure(figsize=(10, 6))
    
    for result in results:
        times, distances = zip(*result['time_series'])
        plt.plot(
            times, 
            distances, 
            label=f"Depth={result['parameters']['max_depth']} (Best: {result['best_distance']:.2f})",
            linewidth=1.5
        )
    
    plt.xlabel("Time (s)")
    plt.ylabel("Best Distance")
    plt.title(
        f"{file_ext} Optimization Progress\n"
        f"n_pop={n_population}, K={K}, μ={mu}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/combined_depth_comparison_{file_ext}.png")
    plt.close()
    
if __name__ == "__main__":
    cords = [[i for i in cord if i] for cord in list(csv.reader(open(filename), delimiter=" "))]
    cities = [City(float(cord[0]), float(cord[1]), i) for i, cord in enumerate(cords)]
    
    distance_matrix = [[0]*len(cities) for _ in range(len(cities))]
    for city1 in cities:
        for city2 in cities:
            distance_matrix[city1.ID][city2.ID] = city1.dist(city2)
    param_grid = {
        'n_population': [50,100,200],
        'K': [2,5,10],
        'mu': [0.05, 0.1],
        'max_depth': [0, 1, 5]
    }
    
    timeout_per_run = 3
    
    import os
    os.makedirs("results", exist_ok=True)
    
    # print("Starting grid search...")
    # results = grid_search(cities, param_grid, timeout_per_run)
    
    # analyze_and_plot_results(results, cities)
    
    # with open('results/grid_search_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)
    
    # print("Grid search completed!")
    
    params = {
        'n_population': 200,
        'K': 10,
        'mu': 0.05,
        'depths': [0, 10, 50],  # Test these depths
        'timeout': timeout,  # seconds per run
        'cities':cities.copy()
    }
    
    print("Running experiments for different depths...")
    results = run_experiment_for_depths(
        params
    )
    
    plot_combined_distance_vs_time(
        results,
        params['n_population'],
        params['K'],
        params['mu']
    )
    
    with open(f'results/depth_comparison_results_{file_ext}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Experiments completed!")