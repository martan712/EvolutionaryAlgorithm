import csv
import math
import functools
import random
import matplotlib.pyplot as plt
from operator import itemgetter
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout

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
        return functools.reduce(lambda s, next: (next, s[1]+s[0].dist(next)), self.cities[1:], (cities[0], 0))[1]
    
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

    def swap_edges(self, i: int, j: int):
        i+=1
        while i < j:
            tmp = self.cities[i]
            self.cities[i] = self.cities[j]
            self.cities[j] = tmp
            i+=1
            j-=1

    def local_ascend(self, max_depth: float):
        found_optimum = True
        while found_optimum and max_depth > 0:
            max_depth -= 1
            found_optimum = False
            for j in range(len(self.cities)-1):
                for i in range(j):
                    lengthDelta: float = self.cities[i].dist(self.cities[j])\
                        + self.cities[i+1].dist(self.cities[(j + 1) % len(self.cities)])\
                            - self.cities[i].dist(self.cities[(i + 1) % len(self.cities)])\
                                - self.cities[j].dist(self.cities[(j + 1) % len(self.cities)]);
                    if lengthDelta < 0:
                        self.swap_edges(i, j)
                        foundImprovement = True
         
    def __str__(self):
        return "->".join(map(lambda city: str(city.ID), self.cities))
    __repr__ = __str__


class Population:
    routes: [Route]

    def __init__(self, cities: [City], N: int):
        self.routes = [] #AAAAAAAA https://stackoverflow.com/questions/3887079/why-does-initializing-a-variable-via-a-python-default-variable-keep-state-across
        cities = cities.copy()
        for i in range(N):
            random.shuffle(cities)
            self.routes.append(Route(cities.copy()))
   
    def tournament_selection(self):
        selected = random.sample(self.routes, K)
        selected.sort(key=lambda x: x.fitness(), reverse=True)
        return selected[0]
    
    def create_new_population(self):
        new_population = []
        while len(new_population) < n_population:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            c1, c2 = p1.crossover(p2)
            new_population.extend([c1, c2])
        self.routes = new_population
        
    def local_search(self, max_depth: float):
        for route in self.routes:
            route.local_ascend(max_depth)

    def mutate(self, mutation_rate: float):
        for route in self.routes:
            if random.uniform(0, 1) < mutation_rate:
                route.mutate()

    def plot_route(self, title="TSPLecture Route", filename=""):
        routes = [(route.fitness(), route) for route in self.routes]
        routes.sort(key=itemgetter(0), reverse=True)
        best_fit, best =routes[0]
        dist = best.distance()
        best = best.cities + [ best.cities[0] ]
        x = [city.x for city in best]
        y = [city.y for city in best]

        plt.figure(figsize=(10, 8), facecolor='white')

        # Plot the route path
        plt.plot(x, y, 'b-', linewidth=1.5, alpha=0.8)

        # Plot the cities
        plt.plot(x, y, 'o', markersize=8, 
                markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.5)

        # Annotate cities
        for i, city in enumerate(best):
            plt.text(city.x, city.y, str(city.ID), 
                    fontsize=9, ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.7, 
                            edgecolor='none', pad=2))

        # Remove axes and spines
        plt.gca().set_facecolor('white')

        # Add title with improved formatting
        plt.title(f"TSP Solution - Total Distance: {dist:.2f}", 
                fontsize=12, pad=20)

        # Equal aspect ratio to prevent distortion
        plt.gca().set_aspect('equal', adjustable='datalim')

        # Tight layout to prevent clipping
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

def run(population: Population, timeout: float, mutation_rate: float, max_depth: float):
    t = []
    score = []
    population.print_best()
    t0 = perf_counter()
    while perf_counter()-t0 < timeout:
        population.create_new_population()
        population.mutate(mutation_rate)
        population.local_search(max_depth)
        
        t.append(perf_counter()-t0)
        score.append(population.get_score())
        print(f"total elapsed time: {perf_counter()-t0}")
    population.print_best()
    population.plot_route(filename=f"tsp_localdepth={max_depth}")
    return t, score

if __name__ == "__main__":
    
    n_population = 200
    K = 5
    mu = 0.1
    # file-tsp
    cords = [[i for i in cord if i] for cord in list(csv.reader(open('file-tsp.txt'), delimiter=" "))]
    cities = [City(float(cord[0]), float(cord[1]), i) for i, cord in enumerate(cords)]

    splt, ax = plt.subplots()
    for max_depth in [0, 1, 5, math.inf]:
        population = Population(cities, n_population)
        t, score = run(population, 3, 0.1, max_depth)
        ax.scatter(t, score)
        ax.scatter(t, score, label=f'max_depth={max_depth}')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fitness")
    ax.legend()
    splt.savefig('figures/file-tsp')


    # #att48.tsp
    # cords = [cord[1:] for cord in list(csv.reader(open('att48.tsp'), delimiter=" "))]
    # cities = [City(float(cord[0]), float(cord[1]), i) for i, cord in enumerate(cords)]

    # splt, ax = plt.subplots()
    # for max_depth in [0, 1, 5, math.inf]:
    #     population = Population(cities, n_population)
    #     t, score = run(population, 400, 0.01, max_depth)
    #     ax.scatter(t, score, label=f'max_depth={max_depth}')
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Fitness")
    # ax.legend()
    # splt.savefig('figures/d1655')


