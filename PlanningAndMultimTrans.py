"""

@author: Leonora Frangu, Fabio James Greenwood, Giada Palma, Benedetta Pasqualetto

"""

### Libraries
import re
import gurobipy as gp
from gurobipy import *
from sklearn.metrics import homogeneity_completeness_v_measure


### Sets and parameters

# Dictionary containing multiple dictionaries of information
input_data = {}

# Open text file for reading data.
txtFile = open('Demo Instances/instance_demo1_N10.txt','r')

# Function definition to transform an element to hours by converting it to integer (eventually back to string)
def to_hours(element):
    h = int(element)/60
    #toStr = str(int(h))    #Convert back to string
    return h


"""
#-------------------------------------------------------- HORIZON -----------------------------------------------
# Dictionary definition of the horizon
horizon_dict = {}

# Function definition to obtain the horizon start and end time
def get_horizon():
    with txtFile as f:
        line = f.readline()
        horizon_dict.update({'start': to_hours(line.split()[1]), 'end': to_hours(line.split()[2])})
    return horizon_dict
#get_horizon() # Check if the horizon is correct

input_data['horizon'] = get_horizon()
#print(input_data)
"""
#----------------------------------------------------------------------------------------------------------------

answer = {}
with txtFile as document:
    for line in document:
        if not re.match("NODE_ID NODE_ID MODE DISTANCE TIME FITNESS",line):
            if line.strip():  # non-empty line?
                key, value = line.split(None, 1)  # None means 'all whitespace', the default
                answer[key] = value.split()
        else: 
            break
print(answer)


def get_horizon():
    answer.keys()


print(get_horizon())
### Initialization of the problem

### Variables

### Constraints

### Objective function

### Resolution
# Calling the solver
# Printing an optimal solution (if it has been found)