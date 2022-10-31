
## Example of a scatter search implementation

### DISTANCE MATRIX
from hashlib import new
import random
import numpy as np
from scipy.spatial import distance

class DistanceMatrixGenerator:
    """ Class responsible for creating a distance matrix between cities """

    def euclideanDistance(self, p1, p2) -> float:
        """ Calculates the Euclidean length between two cities with coordinates (x,y)
            :parameter p1 - [x, y].
            :parameter p2 - [x, y].
        """
        return distance.euclidean(p1, p2)

    def createDistanceMatrix(self, coordinates) -> list:
        """ Function that creates a distance matrix between cities
            e.g. [0, 100, 150]
                [50, 0, 100]
                [12, 20, 0]
            where "0" - denotes the current city, other values - the cost of visiting a given city
        """
        distances = []
        for i in range(0, len(coordinates)):
            temp = []
            for j in range(0, len(coordinates)):
                temp.append(self.euclideanDistance(coordinates[j], coordinates[i]))
            distances.append(temp)
        return distances


### DIVERSIFICATION GENERATION METHOD
import numpy as np

class DiversificationGenerator:
    """ Diversification generator that creates new initial solutions """

    def generate(self, seed: list, startElement: int = 5, n: int = 10) -> list:
        """ Function that generates n solutions in the form of permutations of the seed generated earlier
            :parameter seed - seed
            :parameter startElement - element of the seed from which we start to create a permutation, e.g. 5
            :parameter n - number of solutions to be generated
            The solution was created based on the publication:
            https://www.researchgate.net/publication/221024271_A_Template_for_Scatter_Search_and_Path_Relinking
            https://www.researchgate.net/publication/313096950_Diversification_Methods_for_Zero-One_Optimization
        """
        print("*** Diversification Generator :: start ***")
        g = startElement # e.g. P(5,5) , P(5,4), P(5,3), P(5, 2), P(5, 1)
        results = []
        maximum = max(seed)
        while len(results) < n:
            permutations = []
            iterator = g
            while iterator > 0:
                subpermutations = [iterator]
                element = iterator
                while True:
                    element += g
                    if element > maximum:
                        break
                    subpermutations.append(element)
                # In order to better illustrate the operation, you can uncomment line 205
                # print(f "subpermutation: P({g},{iterator}) - {subpermutations}")
                permutations.append(subpermutations)
                iterator -= 1
            g += 1
            results.append(np.concatenate(permutations))
        print(f"*** Diversification Generator :: end - created {len(results)} solutions ***")
        return results

    def generate2(self, dataset: np.array) -> list:
        """" Function that generates a seed - random path from (1, n) cities
            :parameter dataset - dataset
            :return randomPath
         """
        populationRange = list(range(1, dataset.shape[0] + 1))
        return random.sample(populationRange, dataset.shape[0])



##### IMPROVEMENT METHOD
import copy
import numpy as np

from utils import calculatePathCost


class Improvement:
    def twoOpt(self, distanceMatrix: np.array, pathWithCost: list) -> list:
        """ A 2-opt function that swaps two edges with other edges to create a new cycle
            and trying to reduce the path cost
            :parameter distanceMatrix - neighborhood matrix
            :parameter pathWithCost - path with current cost
            :return bestPath - improved path with improved cost
         """
        bestPath = copy.deepcopy(pathWithCost)
        pathLength = len(bestPath[0])
        currentPath = copy.deepcopy(bestPath)
        resetPathVariable = copy.deepcopy(bestPath)
        for i in range(0, pathLength - 2):
            for j in range(i + 1, pathLength - 1):
                currentPath[0] = self.twoOptSwap(bestPath[0], i, j)
                currentPath[1] = calculatePathCost(distanceMatrix, currentPath[0])
                if currentPath[1] < bestPath[1]:
                    for k in range(pathLength):
                        """ copy city positions from improved path """
                        bestPath[0][k] = currentPath[0][k]
                    bestPath[1] = currentPath[1]
                currentPath = copy.deepcopy(resetPathVariable)
        # print(f "pathWithCost: {pathWithCost} bestPath: {bestPath}")
        return bestPath

    def twoOptSwap(self, path: list, i: int, j: int) -> list:
        """ Function that swaps the order of elements in an array from index i to j """
        swapped = copy.deepcopy(path)
        swapped[i: j + 1] = list(reversed(swapped[i: j + 1]))
        swapped[-1] = swapped[0]
        return swapped



### REFERENCE SET UPDATE 
from typing import List
from time import time


def initialPhase(distancesMatrix: np.array, n: int, b: int, startElement: int) -> List[list]:
    """ Initial phase of the scatter search algorithm
        :parameter distancesMatrix - matrix of distances of cities
        :parameter n - number of initial solutions formed by DG
        :parameter b - number of solutions of RefSet
        :parameter startElement - the element from which the permutations performed on the seed start, e.g. 5 for the version with the permutation generator
        :return RefSet - b of the best initial solutions
    """
    if n < b:
        raise ValueError("b cannot be greater than n!")
    print("*** Initializing phase ***")
    RefSet = []
    DG = DiversificationGenerator()
    startTime = time()
    while len(RefSet) < b:
        """ 1.1 Choose 1 of 2 diversification generators - create initial solutions """
        # diverseTrialSolutions = diversificationGeneratorForRandomPaths(DG, distancesMatrix, n)
        diverseTrialSolutions = diversificationGeneratorForSeedPermutations(DG, distancesMatrix, startElement, n)

        enhancedSolutions = []
        for i in range(len(diverseTrialSolutions)):
            path = diverseTrialSolutions[i]
            pathCost = calculatePathCost(distancesMatrix, path)
            """ 1.2 Improve initial solutions """
            enhancedSolutions.append(improvementFactor.twoOpt(distancesMatrix, [path, pathCost]))

        """ 1.3 Replenish the RefSet with enhanced paths """"
        print(f"Improved {len(enhancedSolutions)} initial solutions")
        [RefSet.append(enhanced) for enhanced in enhancedSolutions]

        if len(RefSet) >= b:
            RefSet.sort(key=lambda x: x[1])
            RefSet = RefSet[:b]

    totalTime = np.around(time() - startTime, 2)
    print(f"RefSet has been filled with {len(RefSet)} solutions - end of initialization phase")
    print(f"Time to generate {b} solutions - {totalTime} sec.")
    return RefSet






def scatterSearch(temporaryRefSet: list, distancesMatrix: np.array, iterations=50) -> tuple:
    """ Implementation of the scatter search algorithm
        :parameter temporaryRefSet - initial RefSet generated in the initialization phase
        :parameter distancesMatrix - neighborhood matrix
        :parameter iterations - number of search iterations
        :returns bestPathWithCost - best path and its cost , costs - cost per iteration, totalTime - total search time
     """
    print("*** Scatter Search :: start ***")
    scm = SolutionCombinationMethod()
    RefSet = copy.deepcopy(temporaryRefSet)
    b = len(RefSet)
    bestPathWithCost = RefSet[0]
    startTime = time()

    costs = []
    index = 0
    for i in range(iterations):
        C = []
        """ 1. crossing elements and improving with 2-opt algorithm """
        for j in range(b):
            C.append(scm.crossover(distancesMatrix, RefSet))
            C[j] = improvementFactor.twoOpt(distancesMatrix, C[j])

        """ 2 Complement the RefSet with the elements of the C array """
        [RefSet.append(c) for c in C]

        """ 3. sorting the RefSet and selecting b best paths """
        RefSet.sort(key=lambda x: x[1])
        RefSet = RefSet[:b]

        """ 4 Update the best path with the lowest cost """
        if RefSet[0][1] < bestPathWithCost[1]:
            bestPathWithCost = RefSet[0]
        index += 1
        costs.append(RefSet[0][1])
        bestPathWithCost[1] = np.around(bestPathWithCost[1], 2)
        print(f"i = {i + 1}/{iterations}, cost = {bestPathWithCost[1]}")
    totalTime = time() - startTime
    print(f"*** Scatter Search :: end - total search time - {totalTime} seconds ***")
    return bestPathWithCost, costs, totalTime







#----------------------------------------------------------------
# pseudocode

# Function to calculate Euclidean length
from scipy.spatial import distance

def euclideanDistance(p1, p2):
    return distance.euclidean(p1, p2)

# Function to create a distance matrix between places, where 0 represents the current place
def createDistanceMatrix(coordinates):
    distances = []
    for i in range(0, len(coordinates)):
        temp = []
        for j in range(0, len(coordinates)):
            temp.append(euclideanDistance(coordinates[j], coordinates[i]))
        distances.append(temp)
    return distances


# Function that generates n solutions in the form of permutations of the seed generated earlier
# seed: matrix of Euclidean distances
# n: number of solutions to be generated
from itertools import permutations 
import random

def generatePermutations(seed, n):
    solutions = []
    for i in n:
        solutions.append(np.concatenate(permutations(seed, n)))
        i+=1
    return solutions

# Function that generates a random path from (1, n) places
# seed: matrix of Euclidean distances
def generateRandomPaths(seed):
    populationRange = list(range(1, seed.shape[0] + 1))
    return random.sample(populationRange, seed.shape[0])


from utils import calculatePathCost

# Function that performs 2-opt local search
def improvementMethodTwoOpt(distanceMatrix, pathWithCost):
    bestPath = pathWithCost
    pathLength = len(bestPath)
    currentPath = bestPath
    resetPathVariable = bestPath

    for i in range(0, pathLength-2):
        for j in range(i+1, pathLength-1):
            # perform the swap between two arcs reducing the circuit length
            currentPath = twoOptSwap(bestPath[0], i, j)
            currentPath = calculatePathCost(distanceMatrix, currentPath)
            if currentPath < bestPath:
                # update
                bestPath = currentPath
        #reset
        currentPath = resetPathVariable

    return bestPath

def twoOptSwap():
    return



# Function to update the reference set of best solutions
# distancesMatrix: matrix of distances of places
# n: number of initial solutions formed by diversification generation method
# b: number of solutions of RefSet
def RefSetUpdate(distancesMatrix, n, b):
    RefSet = []
    while len(RefSet) < b:
        # Create initial solutions
        diverseTrialSolutions = generatePermutations(distancesMatrix, n)
        enhancedSolutions = []

        for i in range(len(diverseTrialSolutions)):
            path = diverseTrialSolutions[i]
            pathCost = calculatePathCost(distancesMatrix, path)
            # Improve initial solutions
            enhancedSolutions.append(improvementMethodTwoOpt(distancesMatrix, pathCost))

        # Populate/Replace the RefSet with enhenced solutions
        for enhanced in enhancedSolutions:
            RefSet.append(enhanced) 

        if len(RefSet) >= b:
            # Sort the RefSet
            RefSet.sort()

    return RefSet


def solutionCombinationMethod():
    return
def subsetGenerationMethod():
    return



def scatterSearch(RefSet, distancesMatrix):


    newSubset = subsetGenerationMethod()
    b = len(RefSet)
    newTrialSol = []
    temporaryRefSet = []

    while newSubset != 0 :
        newSol = False
        
        # apply solution ocmbination method and improvement method
        for j in range(b):
            newTrialSol.append(solutionCombinationMethod(distancesMatrix, RefSet))
            newTrialSol[j] = improvementMethodTwoOpt(distancesMatrix, newTrialSol[j])

        # update reference set
        RefSet.append(RefSetUpdate(distancesMatrix, newTrialSol, b))

        if RefSet != temporaryRefSet:
            newSol = True
        else: 
            break


