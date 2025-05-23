import math

# Coordinates of the cities (from the previous list)
coordinates = [
    (6734, 1453), (2233, 10), (5530, 1424), (401, 841), (3082, 1644), (7608, 4458), 
    (7573, 3716), (7265, 1268), (6898, 1885), (1112, 2049), (5468, 2606), (5989, 2873),
    (4706, 2674), (4612, 2035), (6347, 2683), (6107, 669), (7611, 5184), (7462, 3590), 
    (7732, 4723), (5900, 3561), (4483, 3369), (6101, 1110), (5199, 2182), (1633, 2809),
    (4307, 2322), (675, 1006), (7555, 4819), (7541, 3981), (3177, 756), (7352, 4506),
    (7545, 2801), (3245, 3305), (6426, 3173), (4608, 1198), (23, 2216), (7248, 3779),
    (7762, 4595), (7392, 2244), (3484, 2829), (6271, 2135), (4985, 140), (1916, 1569),
    (7280, 4899), (7509, 3239), (10, 2676), (6807, 2993), (5185, 3258), (3023, 1942)
]

# Tour sequence
# online given
tour = [1, 8, 38, 31, 44, 18, 7, 28, 6, 37, 19, 27, 17, 43, 30, 36, 46, 33, 20, 47, 21, 32, 39, 48, 5, 42, 24, 10, 45, 35, 4, 26, 2, 29, 34, 41, 16, 22, 3, 23, 14, 25, 13, 11, 12, 15, 40, 9,1]

#best found
# tour = [21,2,22,13,24,12,10,11,14,39,8,0,7,37,30,43,17,6,27,5,36,18,26,16,42,29,35,45,32,19,46,20,31,38,47,4,41,23,9,44,34,3,25,1,28,33,40,15,21]
# tour = [x+1 for x in tour]
print(len(tour))
# Function to calculate the Euclidean distance between two points
import math

def att_distance(p1, p2):
    xd = p1[0] - p2[0]
    yd = p1[1] - p2[1]
    
    rij = math.sqrt((xd * xd + yd * yd) / 10.0)
    tij = round(rij)
    
    if tij < rij:
        return tij + 1
    else:
        return tij

# Calculate the total distance
total_distance = 0
for i in range(len(tour) - 1):
    city1 = coordinates[tour[i]-1]
    city2 = coordinates[tour[i + 1]-1]
    total_distance += att_distance(city1, city2)

# Output the total distance
print(total_distance)
