import csv
import math
import functools
import random
from pprint import pprint
from operator import itemgetter
from time import perf_counter

n_population = 50
replacement_pairs = 10
iterations = 50
mutation_rate = 0.1 # change a route has (exactly) one mutations

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

    def fitness(self):
        return 1/functools.reduce(lambda s, next: (next, s[1]+s[0].dist(next)), self.cities[1:], (cities[0], 0))[1]

    def crossover(self, other):
        # Instead we could also sample from 0...len(cities-1), but alas
        start = random.randint(0, len(self.cities))
        end = start
        while end == start:
            end = random.randint(0, len(self.cities))
        if start > end:
            x = end
            end = start
            start = x

        #print(self.cities, other.cities) 
        #print(start, end) 

        middle1 = self.cities[start:end]
        middle2 = other.cities[start:end]
        #print(middle1) 

        sides1 = [city for city in other.cities if not city in middle1]
        sides2 = [city for city in self.cities if not city in middle2]
        
        child1 = middle1 + sides1
        child2 = middle2 + sides2
        #print(child1) 
        child1 = child1[len(child1)-start:]+child1[:len(child1)-start]
        child2 = child2[len(child1)-start:]+child2[:len(child1)-start]
        #print(child1) 
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
 

    def local_ascend(self):
        #print(f"before: {self.fitness()} {self}")
        found_optimum = True
        while found_optimum:
            found_optimum = False
            for j in range(len(self.cities)-1):
                for i in range(j):
                    lengthDelta: float = self.cities[i].dist(self.cities[j]) + self.cities[i+1].dist(self.cities[(j + 1) % len(self.cities)]) - self.cities[i].dist(self.cities[(i + 1) % len(self.cities)]) - self.cities[j].dist(self.cities[(j + 1) % len(self.cities)]);
                    if lengthDelta < 0:
                        self.swap_edges(i, j)
                        foundImprovement = True
                        #print(f"\tmid({lengthDelta}, {i}, {j}): {self.fitness()} {self}")
        #print(f"after: {self.fitness()} {self}")
         
    def __str__(self):
        return "->".join(map(lambda city: str(city.ID), self.cities))
    __repr__ = __str__


class Population:
    routes: [Route] = []

    def __init__(self, cities: [City], N: int):
        cities = cities.copy()
        for i in range(N):
            random.shuffle(cities)
            #print(cities)
            self.routes.append(Route(cities.copy()))
        #print(self.routes)


    def next_gen(self):
        routes = [(route.fitness(), route) for route in self.routes]
        routes.sort(key=itemgetter(0), reverse=True)
        #print(routes)
        routes = [route for fitness, route in routes[:len(routes)-replacement_pairs*2]]
        lucky_ones = random.sample(routes, replacement_pairs*2)

        pairs = zip(lucky_ones[:replacement_pairs], lucky_ones[replacement_pairs:])
        for pair in pairs:
            children = pair[0].crossover(pair[1])
            routes.append(children[0])
            routes.append(children[1])

        self.routes = routes
   
    def local_search(self):
        for route in self.routes:
            route.local_ascend()

    def mutate(self):
        for route in self.routes:
            if random.uniform(0, 1) < mutation_rate:
                route.mutate()

    def print_best(self):
        routes = [(route.fitness(), route) for route in self.routes]
        routes.sort(key=itemgetter(0), reverse=True)
        #print(routes)
        routes = [route for fitness, route in routes]
        print(f"{routes[0].fitness()}: {routes[0]}")

cords = [[i for i in cord if i] for cord in list(csv.reader(open('file-tsp.txt'), delimiter=" "))]
cities = [City(float(cord[0]), float(cord[1]), i) for i, cord in enumerate(cords[:])]

population = Population(cities, n_population)
print("start:")
population.print_best()
t0 = perf_counter()
for i in range(iterations):
    population.next_gen()
    population.mutate()
    population.print_best()
    print(f"total elapsed time: {perf_counter()-t0}")


population = Population(cities, n_population)
print("start:")
population.print_best()
t0 = perf_counter()
for i in range(iterations):
    population.next_gen()
    population.mutate()
    population.local_search()
    population.print_best()
    print(f"total elapsed time: {perf_counter()-t0}")
