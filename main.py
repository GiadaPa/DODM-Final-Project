#!/usr/bin/env python3.7

# Copyright 2022, Gurobi Optimization, LLC

# Solve a traveling salesman problem on a randomly generated set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

print("hello")

import sys
import math
import random
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
from import_function import import_inputs
from run_scenario import run_scenario
from os import listdir
from os.path import isfile, join
exec(open("import_function.py").read())
exec(open("run_scenario.py").read())



explicit_input_folder_location = "C:/Users/fabio/OneDrive/Documents/Studies/Discrete_Optimisation/DODM-Final-Project/Demo Instances/inputs_test/"
explicit_output_folder_location = "C:/Users/fabio/OneDrive/Documents/Studies/Discrete_Optimisation/DODM-Final-Project/Demo Instances/outputs_test/"
input_file_names = [f for f in listdir(explicit_input_folder_location) if isfile(join(explicit_input_folder_location, f))]

#this will loop all the inputs we have put in the folder
for scenario_input_file in input_file_names:
    input_objects, input_links, input_global = import_inputs(explicit_input_folder_location + scenario_input_file)
    run_scenario(input_objects, input_links, input_global)
    #results_string = run_scenario(input_objects, input_links, input_global, solver = solver)
    #create csv output -> explicit_output_folder_location
    
    


print("Hello")
