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
    #the time required to travel down arc ij via mode m
    #Set of all starting points of person n
    const_h = dict()
    const_h_string = "const_h_p_n_[{},{}]"
    for person_id in index_person_ids:
        for node_id in index_nodes_ids:
            const_h[person_id, node_id] = 0
    for person_id in index_person_ids:
        node_id = input_objects["PEOPLE"][person_id]["HOME_ID"]
        const_h[person_id, node_id] = 1
        
        

    #travelling time associated to each arc
    #const_t = gp.tupledict()
    const_t = dict()
    const_t_string = "const_t_i_j_m_[{},{},{}]"
    for i_id, j_id, mode_id in input_links["NODE_TRAVEL_INFO"].keys():
        value = input_links["NODE_TRAVEL_INFO"][i_id, j_id, mode_id]["TIME"]
        #const_t[i_id, j_id, mode_id] = m.addVar(vtype=GRB.CONTINUOUS, lb=value, ub=value, name=const_t_string.format(i_id, j_id, mode_id))
        const_t[i_id, j_id, mode_id] = value
    del value
    m.update()
    
    #const_t_n = gp.tupledict()
    const_t_n = dict()
    const_t_n_string = "const_t_i_j_m_n_[{},{},{},{}]"
    for i_id, j_id, mode_id in input_links["NODE_TRAVEL_INFO"].keys():
        for person_id in index_person_ids:
            value = input_links["NODE_TRAVEL_INFO"][i_id, j_id, mode_id]["TIME"]
            const_t_n[i_id, j_id, mode_id, person_id] = value
            #const_t_n[i_id, j_id, mode_id, person_id] = m.addVar(vtype=GRB.CONTINUOUS, lb=value, ub=value, name=const_t_n_string.format(i_id, j_id, mode_id, person_id))
    del value
    m.update()
    
    
    
    
    
    
    
    

    """Independent Variables"""    
    #whether person n travels down arc ij on tranportation mode m (Bool)
    #Here every combination of two lists need to be produced, to ensure we have a sparse various
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
    
    
    
    """Semi-Dependant Variables"""
    #These are the variables that are technically dependant variables but are modelled as constrained independent variables



    """Dependent Variables (Constraints and Variable Declaration)"""
    #This section is saved for any variable which are fully dependant on other variables, 
    # this is generally used if various variables need to be consolidated into a single figure 
    # i.e., the sum of costs. Please note that when using these variables, the developer needs to
    # be careful not to make any illegal calculations this new variable


    
    """Constraints"""
    #Basic Conservation of Flow - BCoF
    #Flow is directional, multi-medium, multi-flow (person) 
    #[the entries/exits from a node j, needs to be larger than one if there is a task at the node or if it is the home node of the person (h) (Boolean values)]
    # "BCoFO" -> (out)
    constr_BCoFO_string = "const_BCoFO_n_p_[{},{}]"    
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            m.addConstr((x_vars.sum(node_id, "*", "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id) + const_h[person_id, node_id]) >= 0 ), name = constr_BCoFO_string.format(node_id, person_id))
            #m.addConstr((x_vars.sum(node_id, "*", "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id)) >= 0 ), name = "BCoFO") 
    del constr_BCoFO_string
    
    # "BCoFI" -> (In)
    constr_BCoFI_string = "const_BCoFI_n_p_[{},{}]"    
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            m.addConstr((x_vars.sum("*", node_id, "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id) + const_h[person_id, node_id]) >= 0 ), name = constr_BCoFI_string.format(node_id, person_id))
            #m.addConstr((x_vars.sum(node_id, "*", "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id)) >= 0 ), name = "BCoFO") 
    del constr_BCoFI_string
            
    #Currently each node can only be visited once to not interfere with constraints around timing
    # "BCoFOs" -> (out single max)
    constr_BCoFOs_string = "const_BCoFOs_n_p_[{},{}]"    
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            m.addConstr((x_vars.sum(node_id, "*", "*", person_id) <= 1 ), name = constr_BCoFOs_string.format(node_id, person_id))
    del constr_BCoFOs_string
    
    # "BCoFIs" -> (in single max)
    constr_BCoFIs_string = "const_BCoFIs_n_p_[{},{}]"    
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            m.addConstr((x_vars.sum(node_id, "*", "*", person_id) <= 1 ), name = constr_BCoFIs_string.format(node_id, person_id))
    del constr_BCoFIs_string
    
    #Task Timing
    #This controls the time (w) which the task at j starts
    
    node_id = 2
    person_id = 1
    
    
    m.addConstr((x_vars.prod(const_t_n, "*", node_id, "*", person_id)  >= 0 ), name = "Test")
    
    
    print("Stop")
    
    
    
    
    
    
    
    
    
    
    
    
            
            
    """Compilation of model for export (export is used for model interrogation)"""
    m.update()   
    m.write(explicit_output_folder_location + "model_export.lp")
    
    
    
    
    """Model Running"""

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
    
