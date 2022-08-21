"""

@author: Leonora Frangu, Fabio James Greenwood, Giada Palma, Benedetta Pasqualetto

"""

### Libraries
import gurobipy as gp
from gurobipy import *


### Sets and parameters


# Open text file for reading data.
txtFile = open('Demo Instances/instance_demo1_N10.txt','r')


# Function definition to transform an element to hours by converting it to integer (eventually back to string)
def to_hours(element):
    h = int(element)/60
    #toStr = str(int(h))    #Convert back to string
    return h



# Dictionary definition of the horizon
horizon_dict = {}

# Function definition to obtain the horizon start and end time
def horizon():
    with txtFile as f:
        line = f.readline()
        horizon_dict.update({'start': to_hours(line.split()[1]), 'end': to_hours(line.split()[2])})
        print(horizon_dict)

horizon() # Check if the horizon is correct





people = []     #Declare an empty list for the people.

"""
def horizon():
    txtFile.seek(0)              
    with txtFile as f: 
        for line1 in f:                # For each line, stored as myline,
            people.append(line1)           # add its contents to mylines.
    return people[0]                          # Print the list.

print(horizon())
"""


### Initialization of the problem

### Variables

### Constraints

### Objective function

### Resolution
# Calling the solver
# Printing an optimal solution (if it has been found)