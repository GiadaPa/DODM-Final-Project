#!/usr/bin/env python3.7

# Copyright 2022, Gurobi Optimization, LLC

# Solve a traveling salesman problem on a randomly generated set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import sys
import math
import random
import pandas as pd
import numpy as np
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
from os import listdir
from os.path import isfile, join


from import_function import import_inputs

"""Temporary Section"""
#This section of code will allow the user to work on this file running a single experiment
#This is to be deleted later once we change the code to run multiple scenarios

#import import_function and run
exec(open("import_function.py").read())
explicit_input_folder_location = "C:/Users/fabio/OneDrive/Documents/Studies/Discrete_Optimisation/DODM-Final-Project/Demo Instances/inputs_test/"
explicit_output_folder_location = "C:/Users/fabio/OneDrive/Documents/Studies/Discrete_Optimisation/DODM-Final-Project/Demo Instances/outputs_test/"
input_file_names = [f for f in listdir(explicit_input_folder_location) if isfile(join(explicit_input_folder_location, f))]




"""Special Functionality"""
#This section is for specialist functionality required to run optimisation (subtour elimitation, pulling of variables from input file etc)

def filter_list_of_tuples(target_list, target_position, target_value):
    output = list(
        filter(
            lambda tup: tup[target_position] == target_value,
            target_list
        )
    )
    return output
    
    



# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._vars)
        # find the shortest cycle in the selected edge list
        tour = subtour(vals)
        if len(tour) < n:
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(gp.quicksum(model._vars[i, j]
                                     for i, j in combinations(tour, 2))
                         <= len(tour)-1)


# Given a tuplelist of edges, find the shortest subtour
def subtour(vals):
    # make a list of edges selected in the solution
    edges = gp.tuplelist((i, j) for i, j in vals.keys()
                         if vals[i, j] > 0.5)
    unvisited = list(range(n))
    cycle = range(n+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle

def run_scenario(input_objects, input_links, input_global):
    print("Breakpoint")
    """Section to delete START"""
    #This section contains code that can be cut out with time

    # Parse argument
    """if len(sys.argv) < 2:
        print('Usage: tsp.py npoints')
        sys.exit(1)
    n = int(sys.argv[1])

    # Create n random points
    random.seed(1)
    points = [(random.randint(0, 100), random.randint(0, 100)) for i in range(n)]

    # Dictionary of Euclidean distance between each pair of points
    dist = {(i, j):
            math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
            for i in range(n) for j in range(i)}"""

    """Section to delete END"""


    """Begin of model creation """
    m = gp.Model()


    """Indexes"""
    #Definition/notation of indexes (I'm unsure if we have to formally declare indexes) however we should list them here so notation is consistant

    #i,j        -> nodes
    min_node_num = list(input_links["NODE_TRAVEL_INFO"].keys())[0][0]
    max_node_num = list(input_links["NODE_TRAVEL_INFO"].keys())[0][0]
    for node in list(input_links["NODE_TRAVEL_INFO"].keys()):
        if node[0] < min_node_num:
            min_node_num = node[0]
        if node[0] > max_node_num:
            max_node_num = node[0]
        if node[1] < min_node_num:
            min_node_num = node[1]
        if node[1] > max_node_num:
            max_node_num = node[1]
    index_nodes_ids = range(min_node_num, max_node_num + 1)
    #m          -> transportation (m)ode
    #p          -> person
    index_person_ids = input_objects["PEOPLE"].keys()
    #d          -> departure time number d. Each bus stops have a number of departure times,each of which are indexed with a number
    #r          -> bus route number r
    #t          -> task number t
    #bikePeriod -> period time for the bike quantity spaces relaxation assumption

    """Constants"""
    #These don't require notation as they will be pulled directly from the input import objects
    
    #Boolean whether node i is person n's home
    
    #what are all mentioned nodes?, creation of empty dict:
    """ for node_id in index_nodes_ids:
        for people_id in index_person_ids:
            const_h[node_id, people_id] = 0"""
    #Assignment of true values
    
        #const_h[input_objects["PEOPLE"][person_id]["HOME_ID"], input_objects["PEOPLE"][person_id]["PERSON_ID"]] = 1
    const_h = gp.tupledict()
    for people_id in index_person_ids:
        for node_id in index_nodes_ids:
            const_h[people_id, node_id] = m.addVar(vtype=GRB.BINARY, lb=0, ub=0 ,name='const_h_p_n_[%d,%d]'%(people_id, node_id))
    #
    #    const_h[people_id, node_id] 
    m.update()
    for person_id in index_person_ids:
        const_h_string = "const_h_p_n_[{},{}]"
        node_id = input_objects["PEOPLE"][person_id]["HOME_ID"]
        #m.setAttr("UB", m.getVarByName(const_h_string.format(person_id, node_id)), 1)
        #m.setAttr("LB", m.getVarByName(const_h_string.format(person_id, node_id)), 1)
    m.update()

    """Independent Variables"""
    
    print("Break pre indy variables")
    
    #whether person n travels down arc ij on tranportation mode m (Bool)
    list_to_combine = [list(input_links["NODE_TRAVEL_INFO"].keys()), list(index_person_ids)]
    links_person_combination_list_pre = list(itertools.product(*list_to_combine))
    links_person_combination_list = []
    for instance in links_person_combination_list_pre:
        links_person_combination_list = links_person_combination_list + [instance[0] + (instance[1], )]
    link_person_combined_dict = dict.fromkeys(links_person_combination_list)
      
    x_vars = m.addVars(link_person_combined_dict.keys(), vtype=GRB.BINARY, name='x_var') 
    
    #whether person n completes task t at node i
    y_vars = gp.tupledict()
    for tasks_id in input_objects["TASKS"].keys():
        location_id = input_objects["TASKS"][tasks_id]["PLACE_ID"]
        person_id = input_objects["TASKS"][tasks_id]["PERSON_ID"]
        y_vars[location_id, tasks_id, person_id] = m.addVar(vtype=GRB.BINARY, name='y_vars[%d,"_",%d,"_",%d]'%(location_id, tasks_id, person_id))
    
    
    
    
    
    
    print("Break post indy variables")
    
    # Create variables
    #vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    #for i, j in vars.keys():
    #    vars[j, i] = vars[i, j]  # edge in opposite direction

    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))


    """Semi-Dependant Variables"""
    #These are the variables that are technically dependant variables but are modelled as constrained independent variables


    """Dependent Variables (Constraints and Variable Declaration)"""
    #This section is saved for any variable which are fully dependant on other variables, 
    # this is generally used if various variables need to be consolidated into a single figure 
    # i.e., the sum of costs. Please note that when using these variables, the developer needs to
    # be careful not to make any illegal calculations this new variable


    # Add degree-2 constraint

    """Constraints"""
    #Basic Conservation of Flow - BCoF
    #Flow is directional, multi-medium, multi-flow (person) 
    #[the entries/exits from a node j, needs to be larger than one if there is a task at the node or if it is the home node of the person (h) (Boolean values)]
    # "BCoFO" -> (out)
   #based on:
   #m.addConstrs(  vars.sum(i, '*') == 2 for i in range(n))
    """m.addConstrs((x_vars.sum(node_id, "*", "*", person_id) >= 0.5 * (y_vars[node_id, "*", person_id] + const_h[node_id, person_id]) 
                  for node_id in index_nodes_ids for person_id in index_person_ids), name = "BCoFO")    
    """#m.addConstrs((x_vars.sum(node_id, "*", "*", person_id) >= 1 for node_id in index_nodes_ids for person_id in index_person_ids), name = "BCoFO")    
    #m.addConstrs((x_vars.sum(node_id, "*", "*", person_id) >= 1 for node_id in index_nodes_ids for person_id in index_person_ids), name = "BCoFO")    
    
        
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            m.addConstr((x_vars.sum(node_id, "*", "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id) + const_h.sum(node_id, person_id)) >= 0 ), name = "BCoFO")
            #m.addConstr((x_vars.sum(node_id, "*", "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id)) >= 0 ), name = "BCoFO")
            
            
            
    
    
    
    m.update()   
    
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)
    m.write(explicit_output_folder_location + "model_export.lp")
    
    print("Break post constraints")
    
    #m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))
    
    

    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)


    # Optimize model

    m._vars = vars
    m.Params.LazyConstraints = 1
    m.optimize(subtourelim)

    vals = m.getAttr('X', vars)
    tour = subtour(vals)
    assert len(tour) == n

    print('')
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g' % m.ObjVal)
    print('')

"""Temporary section"""
#This is a temporary section to allow for the inporting of inputs and running of the model function (above) 
#for a single run for the proposes of development

explicit_input_folder_location = "C:/Users/fabio/OneDrive/Documents/Studies/Discrete_Optimisation/DODM-Final-Project/Demo Instances/inputs_test/"
explicit_output_folder_location = "C:/Users/fabio/OneDrive/Documents/Studies/Discrete_Optimisation/DODM-Final-Project/Demo Instances/outputs_test/"
input_file_names = [f for f in listdir(explicit_input_folder_location) if isfile(join(explicit_input_folder_location, f))]
input_objects, input_links, input_global = import_inputs(explicit_input_folder_location + input_file_names[0])
run_scenario(input_objects, input_links, input_global)    
    
