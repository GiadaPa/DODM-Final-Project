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


from os import listdir
from os.path import isfile, join
from datetime import datetime
import pathlib



import sys
import math
import random
from numpy.lib import real
import pandas as pd
import numpy as np
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
import copy
import pathlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

from cmath import isnan
import gurobipy as gp
from gurobipy import *
import pandas as pd
import numpy as np
import copy



if len(sys.argv) == 1:
    rum_lim_minutes = 1
else:
    print("S")
    rum_lim_minutes = float(sys.argv[1])


input_request_objects = {
    "PEOPLE" : {
        "PERSON_ID" : "ID",
        "HOME_ID" : "Int",
        "HOME_LAT" : "Float",
        "HOME_LON" : "Float",
        "BUDGET" : "Int",
        "MAX_NB_CHANGES_TRANSPORT" : "Int"},

    "PLACES" : {
        "PLACE_ID" : "ID",
        "PLACE_LAT" : "Float",
        "PLACE_LON" : "Float",
        "MAX_NB_PEOPLE" : "Int"},
    
    "TASKS" : {
        "TASK_ID" : "ID",
        "PERSON_ID" : "Int",
        "PLACE_ID" : "Int",
        "COST" : "Int",
        "SERVICE_TIME" : "Int",
        "START_TIME" : "Int",
        "END_TIME" : "Int",
        "IS_SPECIAL" : "Int",
        "EXTRA_SERVICE_TIME" : "Int",
        "PENALTY" : "Int"},
    
    "BIKE_STATIONS" : {
        "BIKE_STATION_ID" : "ID",
        "BIKE_STATION_LAT" : "Float",
        "BIKE_STATION_LON" : "Float",
        "NB_AVAILABLE_BIKES" : "Int",
        "NB_FREE_SPOTS" : "Int"},
    
    "BUS_LINES" : {
        "LINE_ID" : "ID",
        "START_TIME" : "Int",
        "FREQUENCY" : "Int",
        "MAX_NB_PEOPLE" : "Int"},
    
    "BUS_STOPS" : {
        "BUS_STOP_ID" : "ID",
        "BUS_STOP_LAT" : "Float",
        "BUS_STOP_LON" : "Float"}
    }

input_request_links = {
    "BUS_STOP_TO_LINE" : {
        "BUS_STOP_ID" : "Int",
        "BUS_LINE_ID" : "Int"},
    
    "NODE_TRAVEL_INFO" : {
        "NODE_ID" : "ID",
        "NODE_ID" : "ID",
        "MODE" : "String",
        "DISTANCE" : "Float",
        "TIME" : "Float",
        "FITNESS" : "Float"}
    }

#tuples are used here because they are ordered
input_request_global = {
    "HORIZON" : [
        ("START" , "Int"),
        ("END" , "Int")],
    
    "MODES_OF_TRANSPORTATION" : [
        ("ID" , "String"),
        ("Name", "String")],
    
    "COST_BIKE_PER_MINUT" : "Float",
    "COST_BUS_PER_RIDE" : "Float"  
    
    }
    

# Dictionary definitions
data_inputs = {}
input_objects = {}
input_links = {}
input_global = {}


#update_data_input(data_input_table, value, format)
#def update_data_input(data_input_table, value_name, value, format, object_name = "None", objects = input_objects.keys()):


# Function read all information in the input file
# Method reads first line of each paragraph to understand which class the input belongs to, 
# then uses/references the input_request dictionaries to understand the expect data and format within that paragraph
#def import_inputs(input_objects = input_objects, input_links = input_links, input_global = input_global):
def import_inputs(explicit_input_file_location = 'inputs/instance_demo1_N10.txt'):
    txtFile = open(explicit_input_file_location,'r')
    input_objects = {}
    input_links = {}
    input_global = {}
    
    input_request_names = list(input_request_objects.keys()) + list(input_request_global.keys())
    with txtFile as f:
        lines = f.readlines()
        line_qty = len(lines)
        currently_reading = True
        current_line = 0
        while currently_reading == True:
            
            #terminate loop
            if current_line >= line_qty:
                currently_reading = False
                break
            
            #skip blank lines
            if lines[current_line] == "\n":
                current_line += 1
                continue
                        
            #determine if paragraph is the start of a new date point and act accordingly
            # if input in input_request_links
            if lines[current_line].split()[0] == "BUS_STOP_ID"  and lines[current_line].split()[1] == "BUS_LINE_ID":
                
                current_class_name = "BUS_STOP_TO_LINE"
                input_links[current_class_name] = {}
                reading_current_class = True
                current_line += 1
                
                while reading_current_class == True:
                        
                    #detect if the next class has arrived
                    if lines[current_line].split()[0] in input_request_names or (lines[current_line].split()[0] == "BUS_STOP_ID"  and lines[current_line].split()[1] == "BUS_LINE_ID") or (lines[current_line].split()[0] == "NODE_ID"    and lines[current_line].split()[1] == "NODE_ID"):
                        reading_current_class = False
                        continue
                    
                    link_name = (lines[current_line].split()[0], lines[current_line].split()[1])
                    input_links[current_class_name][link_name] = 0
                    
                    try:
                        if lines[current_line].split()[2] == "(DEPOSIT)":
                            input_links[current_class_name][link_name] = 1
                    except IndexError as err:
                        a = 1
                        del a
                
                    current_line += 1
                    
                    if lines[current_line] == "\n":
                        reading_class_instance = False
                        current_line += 1
                        break
                
            elif lines[current_line].split()[0] == "NODE_ID"    and lines[current_line].split()[1] == "NODE_ID":
                current_class_name = "NODE_TRAVEL_INFO"
                input_links[current_class_name] = {}
                reading_current_class = True
                current_line += 1
                while reading_current_class == True:
                        
                    #detect if the next class has arrived
                    if lines[current_line].split()[0] in input_request_names or (lines[current_line].split()[0] == "BUS_STOP_ID"  and lines[current_line].split()[1] == "BUS_LINE_ID") or (lines[current_line].split()[0] == "NODE_ID"    and lines[current_line].split()[1] == "NODE_ID"):
                        reading_current_class = False
                        continue
                
                    link_ID = (int(lines[current_line].split()[0]),int(lines[current_line].split()[1]),lines[current_line].split()[2])
                    input_dict = { "DISTANCE" : float(lines[current_line].split()[3]), "TIME" : float(lines[current_line].split()[4]), "FITNESS" : float(lines[current_line].split()[5])}
                    input_links[current_class_name][link_ID] = input_dict
                    
                    current_line += 1
                    if current_line >= line_qty:
                        reading_current_class = False
                        currently_reading = False
                        break
                    
                    
                    if lines[current_line] == "\n":
                        reading_class_instance = False
                        currently_reading == False
                        current_line += 1
                        break
                
                
            elif lines[current_line].split()[0] == "MODES_OF_TRANSPORTATION":
                
                current_class_name = copy.deepcopy(lines[current_line].split()[0])
                input_global[current_class_name] = {}
                reading_current_class = True
                ID = 1
                current_line += 1
                
                while reading_current_class == True:
                        
                    #detect if the next class has arrived
                    if lines[current_line].split()[0] in input_request_names or (lines[current_line].split()[0] == "BUS_STOP_ID"  and lines[current_line].split()[1] == "BUS_LINE_ID") or (lines[current_line].split()[0] == "NODE_ID"    and lines[current_line].split()[1] == "NODE_ID"):
                        reading_current_class = False
                        continue
                    
                    input_global[current_class_name][ID] = lines[current_line].split()[1]
                    ID += 1
                    current_line += 1
                    
                    if lines[current_line] == "\n":
                        reading_class_instance = False
                        current_line += 1
                        break                 
                    
                ID = np.nan
                                    
                print("Hello")
                    
            #if not in input_request_links
            elif lines[current_line].split()[0] in input_request_names:
                # if input in input_request_objects
                if lines[current_line].split()[0] in input_request_objects.keys():
                    
                    current_class_name = copy.deepcopy(lines[current_line].split()[0])
                    input_class = input_request_objects[lines[current_line].split()[0]]
                    reading_current_class = True
                    input_objects[lines[current_line].split()[0]] = {}
                    current_line += 1
                    
                    #loop for reading all values of class X                    
                    while reading_current_class == True:
                        
                        #detect if the next class has arrived
                        
                        if lines[current_line].split()[0] in input_request_names or (lines[current_line].split()[0] == "BUS_STOP_ID"  and lines[current_line].split()[1] == "BUS_LINE_ID") or (lines[current_line].split()[0] == "NODE_ID"    and lines[current_line].split()[1] == "NODE_ID"):
                            reading_current_class = False
                            continue
                        
                        #detect if there is an ID for class
                        try:
                            if input_class[lines[current_line].split()[0]] == "ID":
                                ID = int(lines[current_line].split()[1])
                            else:
                                ID = np.nan    
                        except IndexError as err:
                            ID = np.nan
                        
                        #read instance of class
                        reading_class_instance = True
                        while reading_class_instance == True:
                            #check for end of class instance
                            if lines[current_line] == "\n":
                                reading_class_instance = False
                                current_line += 1
                                continue
                            
                            input_objects = update_input_table(input_objects, input_value=lines[current_line].split()[1], input_name=lines[current_line].split()[0], input_format = input_class[lines[current_line].split()[0]], object_class_name=current_class_name , ID = ID)
                            current_line += 1
                        
                # if input in input_request_global
                elif lines[current_line].split()[0] in input_request_global.keys():
                    input_class = input_request_global[lines[current_line].split()[0]]
                    if not isinstance(input_class, list):
                        #input_global[lines[current_line].split()[0]] = lines[current_line].split()[1]
                        input_global     = update_input_table(input_global, input_value=lines[current_line].split()[1], input_name=lines[current_line].split()[0], input_format = input_class)
                    else:
                        for key, i in zip(input_class, range(1, len(input_class)+1)):
                            #input_global[key] = lines[current_line].split()[i]
                            input_global = update_input_table(input_global, input_value=lines[current_line].split()[i], input_name=key[0], input_format = key[1])
                    current_line += 1    
            
            else:
                current_line += 1
                            
    return input_objects, input_links, input_global
    

                   
def update_input_table(input_table, input_value, input_name, input_format, object_class_name = np.nan, ID = np.nan):
    
    #create new object instance if required for object
    if not pd.isnull(ID) and not ID in input_table[object_class_name]:
        input_table[object_class_name][ID] = {}
        
    if input_format == "Int":
        input_var = int(input_value)
    elif input_format == "ID":
        input_var = int(input_value)
    elif input_format == "Float":
        input_var = float(input_value)
    elif input_format == "String":
        input_var = input_value
    else:
        raise Exception("Error: format " + input_format + " not found")
    
    if pd.isnull(ID):
        input_table[input_name] = input_var
    else:
        input_table[object_class_name][int(ID)][input_name] = input_var
    
    return input_table
                    



"""Temporary Section"""
#This section of code will allow the user to work on this file running a single experiment
#This is to be deleted later once we change the code to run multiple scenarios

#import import_function and run

print("Start at: " + str(datetime.now()))
if len(sys.argv) == 1:
    explicit_input_folder_location  = str(pathlib.Path(__file__).parent.resolve()) + "\\inputs\\"
    explicit_output_folder_location = str(pathlib.Path(__file__).parent.resolve()) + "\\outputs\\"
else:
    explicit_input_folder_location  = str(pathlib.Path(sys.argv[0]).parent) + "\\inputs\\"
    explicit_output_folder_location = str(pathlib.Path(sys.argv[0]).parent) + "\\outputs\\"
input_file_names = [f for f in listdir(explicit_input_folder_location) if isfile(join(explicit_input_folder_location, f))]


"""Special Functionality"""
#This section is for specialist functionality required to run optimisation (subtour elimitation, pulling of variables from input file etc)

def save_values(input_objects, input_links, input_global, output_people_routes, output_people_route_methods, output_people_route_times, model):
    
    #global input_global
    #global input_objects
    #global input_links
    #global output_people_routes
    #global output_people_route_methods
    #global output_people_route_times
    
    save_file = {
        "input_global" : input_global,
        "input_objects" : input_objects,
        "input_links" : input_links,
        "output_people_routes" : output_people_routes,
        "output_people_route_methods" : output_people_route_methods,
        "output_people_route_times" : output_people_route_times#,
        #"model" : model
    }
    
    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(save_file, f)
        

def load_values():
    with open('saved_dictionary.pkl', 'rb') as f:
        save_file = pickle.load(f)
    return save_file

    

def filter_list_of_tuples(target_list, target_position, target_value):
    output = list(
        filter(
            lambda tup: tup[target_position] == target_value,
            target_list
        )
    )
    return output
    

def return_unique_values_in_tuples(input_tuples):
    output = []
    for i in input_tuples:
        for j in i:
            if not j in output:
                output = output + [j]
    return output
    
    
    

def return_if_valid_reference(matrix, reference, output_if_false=0, output_if_true="value"):
    
    
    try:
        a               = matrix[tuple(reference)]#matrix[reference]
        true_reference  = True
    except KeyError:
        a               = False
        true_reference  = False
       
    if true_reference == True   and output_if_true == "value":
        return a
    if true_reference == True   and output_if_true != "value":
        return output_if_true
    if true_reference == False:
        return output_if_false
    
def are_there_tasks_with_the_same_person_and_node():    
    duplicate = False
    for task_ids_a in input_objects["TASKS"].keys():
        task_a      = input_objects["TASKS"][task_ids_a]
        #person_id   = task_a["PERSON_ID"]
        #place_id    = task_a["PLACE_ID"]
        for task_ids_b in input_objects["TASKS"].keys():
            task_b  = input_objects["TASKS"][task_ids_b]
            if task_ids_a != task_ids_b:
                if task_a["PERSON_ID"] == task_b["PERSON_ID"] and task_a["PLACE_ID"] == task_b["PLACE_ID"]:
                    duplicate = True
    return duplicate
    
        
def return_combinations_for_a_tour():
    
    
    
    return "dd"


# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        vals            = model.cbGetSolution(model._x_vars)
        index_nodes_ids = model._index_nodes_ids
        input_objects   = model._input_objects  #actions, this can be slimmed down
        index_person_ids= model._index_person_ids
        #per person
        for n_ in list(index_person_ids): #action fix this
            # find the shortest cycle in the selected edge list
            tour, is_sub_tour_detected = subtour(vals, n_, index_nodes_ids, input_objects)
            if is_sub_tour_detected == True:
                # add subtour elimination constr. for every pair of cities in tour
                #model.cbLazy(gp.quicksum(model._vars[i, j]
                #                        for i, j in combinations(tour, 2))
                #            <= len(tour)-1)
                for n__ in index_person_ids:
                    links = [(i, j, m, n3) for i,j,m,n3 in vals.keys() if n3 == n__ and i in tour and j in tour]
                    model.cbLazy(gp.quicksum(model._x_vars[i4, j4, m4, n4]
                                            for i4, j4, m4, n4 in links)
                                <= len(tour)-1)

"""

"""


# Given a tuplelist of edges, find the shortest subtour, for person n
def subtour(vals, n, index_nodes_ids, input_objects):
    
    #home = input_objects["PEOPLE"][n]["HOME_ID"]
    # make a list of edges selected in the solution
    edges = gp.tuplelist((i, j) for i,j,m,n_ in vals.keys() if vals[i,j,m,n] > 0.5 and n == n_ )
    
    unvisited = return_unique_values_in_tuples(edges)
    cycle = range(max(index_nodes_ids)+2)  # initial length has 1 more city
    #[key[2] for key in DTime.keys() if (key[0] == l and key[1] == i)]
    #cycle = list(index_nodes_ids) + [max(index_nodes_ids) + 1]  # initial length has 1 more city
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
    if len(cycle) < len(return_unique_values_in_tuples(edges)):
        is_sub_tour_detected = True
    else:
        is_sub_tour_detected = False

    return cycle, is_sub_tour_detected

def run_scenario(input_objects, input_links, input_global, scenario_name = "instance_demo1_N10", rum_lim_minutes = 3, disable_costly_constraints = False, force_1_to_catch_a_bus = False, show_fig = False):

    
    """Begin of model creation """
    md = gp.Model()
    are_there_tasks_with_the_same_person = are_there_tasks_with_the_same_person_and_node()

    """Indexes"""
    #Definition/notation of indexes (I'm unsure if we have to formally declare indexes) however we should list them here so notation is consistant

    # n ∈ N → (person"/" people )
    index_person_ids = input_objects["PEOPLE"].keys()
    # t ∈ T → Tasks  
    index_task_ids = input_objects["TASKS"].keys()
    # h ∈ H → Home nodes
    index_person_ids = input_objects["PEOPLE"].keys()
    # p ∈ P → Place nodes
    index_place_ids = input_objects["PLACES"].keys()
    # b ∈ B → Bike Station nodes
    index_bike_stations_ids = input_objects["BIKE_STATIONS"].keys()
    # s ∈ S → Bus Stops 
    index_bus_stops_ids = input_objects["BUS_STOPS"].keys()
    # l ∈ L → Bus Lines
    index_bus_lines_ids = input_objects["BUS_LINES"].keys()
    # A := H ∪ P ∪ B ∪ S
    index_nodes_ids = list(input_objects["PEOPLE"].keys()) + list(index_place_ids) + list(index_bike_stations_ids) + list(index_bus_stops_ids)
    # i, j ∈ A → nodes 
    #not needed
    # m ∈ M → transportation (m)ode
    index_modes_of_transport = ["WALKING", "CYCLING", "BUS"]


    """Special Subsets"""
    # t ∈ T_(i,n) → Tasks related to node i and person n
    index_subset_tasks_in = dict()
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            index_subset_tasks_in[node_id, person_id] = []
    for task_id in input_objects["TASKS"].keys():
        node_id     = input_objects["TASKS"][task_id]["PLACE_ID"]
        person_id   = input_objects["TASKS"][task_id]["PERSON_ID"]
        task_id     = input_objects["TASKS"][task_id]["TASK_ID"]
        index_subset_tasks_in[node_id, person_id] = index_subset_tasks_in[node_id, person_id] + [task_id]
    
    # i ∈ V_l → nodes along route l
    route_lnum = []
    BUS_STOP_TO_LINE = input_links["BUS_STOP_TO_LINE"]
    NODE_TRAVEL_INFO = input_links["NODE_TRAVEL_INFO"]
    for l in input_objects["BUS_LINES"].keys():
        route_lnum_single   = []
        nodes_unordered     = [int(key[0]) for key in BUS_STOP_TO_LINE.keys() if int(key[1]) == l]
        depot_id            = {key: value for key, value in zip(BUS_STOP_TO_LINE.keys(), BUS_STOP_TO_LINE.values()) if int(key[1]) == l and value == 1}
        depot_id            = int(list(depot_id.keys())[0][0])
        route_lnum_single   = [depot_id]
        for num in range(0, len(nodes_unordered)-1):
            origin              = route_lnum_single[-1]
            links_shortlist_a   = [key[0] for key in NODE_TRAVEL_INFO.keys() if ((int(key[0]) == origin or int(key[1]) == origin) and key[2] =="BUS")]
            links_shortlist_b   = [key[1] for key in NODE_TRAVEL_INFO.keys() if ((int(key[0]) == origin or int(key[1]) == origin) and key[2] =="BUS")]
            links_shortlist     = links_shortlist_a + links_shortlist_b
            node                = [node for node in links_shortlist if ((node in nodes_unordered) and (not node in route_lnum_single))]
            if len(node) > 1:
                raise Exception("Error check interaction between bus route input and model")
            route_lnum_single   = route_lnum_single + node
        route_lnum = route_lnum + [route_lnum_single]
    
    
    # (n,i) ∈ Home → Home node i of each person
    subset_Home_ni = []
    for person in input_objects["PEOPLE"].values():
        location_id = person["HOME_ID"]
        person_id   = person["PERSON_ID"]
        subset_Home_ni = subset_Home_ni + [(person_id, location_id)]
    
    # i ∈ route_l → nodes line l travels down
    
    
    # d ∈ departures_(l,i) → departure d for line l at node i
    
    
    # (t,i,n) ∈ task_details_list_()
    index_task_details_list = []
    for task in input_objects["TASKS"].values():
        t = task["TASK_ID"]
        i = task["PLACE_ID"]
        n = task["PERSON_ID"]
        index_task_details_list = index_task_details_list + [(t,i,n)]
    
    # (t,i) ∈ personal_task_(n)
    index_personal_tasks = dict()
    for n in index_person_ids:
        index_personal_tasks_single = []
        for task in input_objects["TASKS"].values():
            if task["PERSON_ID"] == n:
                index_personal_tasks_single += [(task["TASK_ID"], task["PLACE_ID"])]
        index_personal_tasks[n] = index_personal_tasks_single
    
    
    """Constants"""
    
    #cost of a task
    #task time window
    const_a_t = dict()
    const_b_t = dict()
    const_c_t = dict()
    for task in input_objects["TASKS"].values():
        const_a_t[task["TASK_ID"]] = task["START_TIME"]
        const_b_t[task["TASK_ID"]] = task["END_TIME"]
        const_c_t[task["TASK_ID"]] = task["COST"]
    md.update()
    

    #task duration
    const_s_t = dict() #
    for task in input_objects["TASKS"].values():
        const_s_t[task["TASK_ID"]] = task["SERVICE_TIME"]
    md.update()
    

    #additional time required for special task* if not done within allotted time window
    """Action: I will have to put a methanium to enforce what is in the * set"""
    const_st_istar = dict() #
    for task in input_objects["TASKS"].values():
        const_st_istar[task["TASK_ID"]] = task["EXTRA_SERVICE_TIME"]
    md.update()

    #latitude and longitude of a place

    #number of bikes available and free spots at a bike station

    #latitude and longitude of a bike station

    #cost of bikes in Verona per minute (see assumptions)

    #latitude and longitude of a bus stop

    #cost of chosing bus as transportation mode
    #not in use, referanced directly

    #set of bus departure times (line l, node i, departure time d)
    start_time  = input_global["START"]
    end_time    = input_global["END"]
    route_time_delay = []
    for line in route_lnum:
        route_time_delay_single = [0]
        for i in range(0,len(line)-1):
            time_delay = route_time_delay_single[-1] + [values["TIME"] for key, values in zip(NODE_TRAVEL_INFO.keys(), NODE_TRAVEL_INFO.values()) if (int(key[0]) == line[i] and int(key[1]) == line[i+1] and key[2] =="BUS")][0]
            route_time_delay_single = route_time_delay_single + [time_delay]
        route_time_delay = route_time_delay + [route_time_delay_single]
    
    DTime = dict()
    for l in index_bus_lines_ids:
        freq = input_objects["BUS_LINES"][1]["FREQUENCY"]
        for i in route_lnum[l-1]:
            route_position  = route_lnum[l-1].index(i)
            current_time    = start_time + route_time_delay[l-1][route_position]
            d = 0
            while current_time < end_time:
                DTime[l,i,d] = current_time
                d += 1
                current_time += freq
    del route_position


    #time window a person has to board a bus (see assumptions)
    bus_relaxation = 2 #minutes

    #Maximum number of people allowed on a bus (see assumptions)

    #Penalty for a task t not performed
    unfinished_task_penalty = 1000
    
    #Max number of times a person n can change transportation mode

    #Boolean whether node i requests service from person n

    #Boolean whether node i is person n's home

    #Set of all starting points of person n
    #Not in use
        
    #travelling time associated to each arc
    const_t_ijm = dict()
    for i_id, j_id, mode_id in input_links["NODE_TRAVEL_INFO"].keys():
        value = input_links["NODE_TRAVEL_INFO"][i_id, j_id, mode_id]["TIME"]
        const_t_ijm[i_id, j_id, mode_id] = value
        const_t_ijm[j_id, i_id, mode_id] = value
    del value
    md.update()

    
    #whether a task is special (can be delayed)
    const_special_t = dict() #
    for task in input_objects["TASKS"].values():
        const_special_t[task["TASK_ID"]] = task["IS_SPECIAL"]
    md.update() 
    
    #fitness coef for a given arc
    const_fitness_ijm = dict()
    for m in index_modes_of_transport:
        for i in index_nodes_ids:
            for j in index_nodes_ids:
                if i != j:
                    value = input_links["NODE_TRAVEL_INFO"][i_id, j_id, mode_id]["FITNESS"]
                    const_fitness_ijm[i,j,m] = value
                    const_fitness_ijm[j,i,m] = value

    #Start of the day (mins)
    #No variable declared, taken from the input objects
    
    #End of the day (mins)
    #No variable declared, taken from the input objects
    
    #weighting for fitness coefficient
    fitness_weighting = 0.1
    
    """M (large) Constants"""
    M_time = input_global["END"] * 100
    

    """Independent Variables"""
    
    
    #whether person n travels down arc ij on tranportation mode m (Bool)
    x_vars = gp.tupledict()
    x_var_string = "x_var_i{}j{}m{}__n{}"
    for i,j,m in input_links["NODE_TRAVEL_INFO"].keys():
        for n in index_person_ids:
            x_vars[i,j,m,n] = md.addVar(vtype=GRB.BINARY, name=x_var_string.format(i,j,m,n))
            if m != "BUS":
                x_vars[j,i,m,n] = md.addVar(vtype=GRB.BINARY, name=x_var_string.format(j,i,m,n))
    
    
    
    #whether person n completes task t at node i
    y_vars = gp.tupledict()
    y_var_string = "y_var_i{}_t{}_n{}"
    for tasks_id in input_objects["TASKS"].keys():
        location_id = input_objects["TASKS"][tasks_id]["PLACE_ID"]
        person_id = input_objects["TASKS"][tasks_id]["PERSON_ID"]
        y_vars[location_id, tasks_id, person_id] = md.addVar(vtype=GRB.BINARY, name=y_var_string.format(location_id, tasks_id, person_id))
    
    #whether person n visits node z
    z_vars = gp.tupledict()
    z_var_string = "z_var_i{}__n{}"
    for i in index_nodes_ids:
        for n in index_person_ids:
            z_vars[i, n] = md.addVar(vtype=GRB.BINARY, name=z_var_string.format(i, n))
        
    
    
    #Boolean,True if task is not done outside allotted time
    #Constrained to zero for tasks where this isn^' t an option
    #these tasks suffer extended times
    tstar_vars = gp.tupledict()
    tstar_var_string = "tstar_var_t{}"
    for tasks_id in input_objects["TASKS"].keys():
        tstar_vars[tasks_id] = md.addVar(vtype=GRB.BINARY, name=tstar_var_string.format(tasks_id))
    
    
    #quantity of bikes available at node i at the end of period i
    
    
    #quantity of bike spaces available at node i at the end of period i
    
    
    #whether fare for bike is incurred at node i
    
    
    #whether person n leaves node i via line l at departure number d
    bus_catch_vars = gp.tupledict()
    bus_catch_var_string = "bus_catch_vars_l{}i{}d{}__n{}"
    for l in index_bus_lines_ids:
        for i in route_lnum[l-1]:
            departure_qty = max([key[2] for key in DTime.keys() if (key[0] == l and key[1] == i)])
            for d in range(0, departure_qty):
                for n in index_person_ids:
                    bus_catch_vars[l, i, d, n] = md.addVar(vtype=GRB.BINARY, name=bus_catch_var_string.format(l, i, d, n))
    del departure_qty
    
    #whether fare for bus is incurred for node i, n person
    fee_bus_vars = gp.tupledict()
    fee_bus_var_string = "fee_bus_vars_i{}__n{}"
    for i in index_bus_stops_ids:
        for n in index_person_ids:
            fee_bus_vars[i,n] = md.addVar(vtype=GRB.BINARY, name=fee_bus_var_string.format(i,n))
    
        
    #health or loss gain according to transportation mode chosen
    
    
    #waiting idle time at node (pre-task)
    aw_vars = gp.tupledict()
    aw_var_string = "aw_vars_i{}__n{}"
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            aw_vars[node_id, person_id] = md.addVar(vtype=GRB.CONTINUOUS, lb=0, name=aw_var_string.format(node_id, person_id))
    
        
    #waiting idle time at node (post-task)
    bw_vars = gp.tupledict()
    bw_var_string = "bw_vars_i{}__n{}"
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            bw_vars[node_id, person_id] = md.addVar(vtype=GRB.CONTINUOUS, lb=0, name=bw_var_string.format(node_id, person_id))
    
    #Amount of money on person n
    #referenced directly
    
    
    """Semi-Dependant Variables"""
    #These are the variables that are technically dependant variables but are modelled as constrained independent variables
    w_vars = gp.tupledict()
    w_var_string = "w_var_i{}_n{}"
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            w_vars[node_id, person_id] = md.addVar(vtype=GRB.CONTINUOUS, lb=0, name=w_var_string.format(node_id, person_id))

    ts_istar_vars = gp.tupledict()
    ts_istar_string = "ts_istar_t{}"
    for t in index_task_ids:
        ts_istar_vars[t] = md.addVar(vtype=GRB.BINARY, name=ts_istar_string.format(t))



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
    constr_BCoFO_string = "const_BCoFO_np[{},{}]"    
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            md.addConstr((x_vars.sum(node_id, "*", "*", person_id) >= 0.5 * z_vars[node_id, person_id]), name = constr_BCoFO_string.format(node_id, person_id))
            #md.addConstr((x_vars.sum(node_id, "*", "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id)) >= 0 ), name = "BCoFO") 
    del constr_BCoFO_string
    
    # "BCoFI" -> (In)
    constr_BCoFI_string = "const_BCoFI_np[{},{}]"    
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            md.addConstr((x_vars.sum("*", node_id, "*", person_id) >= 0.5 * z_vars[node_id, person_id]), name = constr_BCoFI_string.format(node_id, person_id))
            #md.addConstr((x_vars.sum(node_id, "*", "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id)) >= 0 ), name = "BCoFO") 
    del constr_BCoFI_string
            
    #Currently each node can only be visited once to not interfere with constraints around timing
    # "BCoFOs" -> (out single max)
    constr_BCoFOs_string = "const_BCoFOs_np[{},{}]"    
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            md.addConstr((x_vars.sum(node_id, "*", "*", person_id) <= 1 ), name = constr_BCoFOs_string.format(node_id, person_id))
    del constr_BCoFOs_string
    
    # "BCoFIs" -> (in single max)
    constr_BCoFIs_string = "const_BCoFIs_np[{},{}]"    
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            md.addConstr((x_vars.sum(node_id, "*", "*", person_id) <= 1 ), name = constr_BCoFIs_string.format(node_id, person_id))
    del constr_BCoFIs_string
    
    #The person must exit their home on foot
    # "BCoF2"
    constr_BCoF2_string = "const_BCoF2_in[{},{}]"    
    for (i, n) in subset_Home_ni:
        md.addConstr((x_vars.sum(i,"*","WALKING", n) == 1), name = constr_BCoF2_string.format(n, i))
    del constr_BCoF2_string
    
    #The person must exit a node they visit
    # "BCoF3"
    constr_BCoF3_string = "const_BCoF3_in[{},{}]"    
    for j in index_nodes_ids:
        for n in index_person_ids:
            md.addConstr((x_vars.sum("*", j, "*", n) == x_vars.sum(j, "*", "*", n)), name = constr_BCoF3_string.format(j, n))
    del constr_BCoF3_string
    
    print("Costly constraint at: " + str(datetime.now()))
    #Task Timing
    #This controls the time (w) which the person arrives at node starts
    #This doesn’t apply where a person is returning to their home node|
    #TT1_ijntm
    constr_TT1_name_string =  "TT1_i{}j{}n{}m{}"
    if disable_costly_constraints == False:
        temp_len = len(index_nodes_ids)
        for i in index_nodes_ids:
            #print(str(i) + "/" +  str(temp_len))
            for j in index_nodes_ids:
                for n in index_person_ids:
                    for m in index_modes_of_transport:
                        if i != j and return_if_valid_reference(x_vars, [i, j, m, n], False, True) and not (n, i) in subset_Home_ni:
                            expr_a_temp = w_vars[i, n] + aw_vars[j, n] + bw_vars[i, n]
                            expr_b_temp = x_vars[i, j, m, n] * const_t_ijm[i,j,m]
                            #These expresions only count if there is a task for the person/node/task combination
                            expr_c_temp = 0
                            for t in index_subset_tasks_in[i,n]:
                                expr_c_temp = expr_c_temp + (const_s_t[t] * y_vars[i, t, n] + const_st_istar[t] * ts_istar_vars[t])
                                
                            expr_d_temp = M_time * (1 - x_vars[i,j,m,n])
                            
                            
                            md.addConstr((w_vars[j, n] >= expr_a_temp + expr_b_temp + expr_c_temp - expr_d_temp), name = constr_TT1_name_string.format(i,j,n,m))
                            
                            """Do not delete, this was the previous requirement to make the constraint, but is still a good example to keep"""
                            """expression_temp = 0
                            for j in index_nodes_ids:
                                for m in index_modes_of_transport:
                                    if return_if_valid_reference(x_vars, [i, j, m, n], False, True):
                                        expression_temp += x_vars[i, j, m, n] * const_t_ijm[i,j,m] 
                            md.addConstr((expression_temp>=1), name = "Test2")"""

                            #md.addConstr((sum(x_vars[i, j, m, n] * const_t_ijm[i,j,m] for j in index_nodes_ids for m in index_modes_of_transport if return_if_valid_reference(x_vars, [i, j, m, n], False, True))>1), name = "Test2")
        del expr_a_temp, expr_b_temp, expr_c_temp, expr_d_temp
    print(datetime.now())
    md.update()
    md.write(explicit_output_folder_location + "model_export.lp")
    
    #Task Timing (cont)
    #This controls that tasks happen within their designated time windows (only applies if the tasks happens 
    #and the task extension penalty for falling out of the allotted time isn’t applied)
    #TT2before_int & TT2after_int
    #also:
    #Task extensions/delays only apply to tasks that are undertaken
    #TT3_int
    
    TT2after_name_string =  "TT2after_i{}n{}t{}"
    TT2before_name_string =  "TT2before_i{}n{}t{}"
    TT3_name_string =  "TT3_i{}n{}t{}"
    for i in index_nodes_ids:
        for n in index_person_ids:
            for t in index_task_ids:
                if return_if_valid_reference(y_vars, [i, t, n], False, True):
                    md.addConstr((const_a_t[t] - M_time * tstar_vars[t] - M_time * (1 - y_vars[i, t, n]) <= w_vars[i, n] + aw_vars[i, n]), name = TT2after_name_string.format(i,n,t))
                    md.addConstr((const_b_t[t] + M_time * tstar_vars[t] + M_time * (1 - y_vars[i, t, n]) >= w_vars[i, n] + aw_vars[i, n]), name = TT2before_name_string.format(i,n,t))
                    #Note: Not sure the below constraint is needed
                    md.addConstr((tstar_vars[tasks_id] <= y_vars[i, t, n]), name = TT3_name_string.format(i,n,t))
    
    #Task extensions/delays only apply to tasks that are special
    #TT4_t
    TT4_name_string =  "TT4_t{}"
    for t in index_task_ids:
        md.addConstr((tstar_vars[tasks_id] <= const_special_t[t]), name = TT4_name_string.format(t))
    
    #A person will only complete a single task at a node they visit
    #TT5_t_n
    TT5_name_string =  "TT5_i{}n{}"
    for i in index_nodes_ids:
        for n in index_person_ids:
            md.addConstr((y_vars.sum(i, "*", n) <= z_vars[i,n]), name = TT5_name_string.format(i,n))
    
    #A person begins at home at the start of the day
    #TT6_in
    TT6_name_string =  "TT6_i{}n{}"
    for (i,n) in subset_Home_ni:
        md.addConstr((w_vars[i, n] >= input_global["START"]), name = TT6_name_string.format(i,n))
    
    #A person leaves their home sometime after the start of the day
    #TT7_in
    TT7_name_string =  "TT7_n{}i{}j{}m{}"
    for (n,i) in subset_Home_ni:
        for j in index_nodes_ids:
            for m in index_modes_of_transport:
                if i != j and return_if_valid_reference(x_vars, [i, j, m, n], False, True):
                    expr_a_temp = input_global["START"] + aw_vars[i, n] + bw_vars[i, n] + x_vars[i, j, m, n] * const_t_ijm[i,j,m]
                    expr_b_temp = 0
                    for t in index_subset_tasks_in[i,n]:
                        expr_b_temp = expr_b_temp + (const_s_t[t] * y_vars[i, t, n] + const_st_istar[t] * ts_istar_vars[t])
                    expr_M_temp = M_time * (1 - x_vars[i,j,m,n])
                    
                    md.addConstr((w_vars[j,n] >= expr_a_temp + expr_b_temp - expr_M_temp), name = TT7_name_string.format(n,i,j,m))
            
    #All time variables must be within the bounds of the day
    #TT8_after_start
    #TT8_before_end
    TT8_after_start = "TT8_after_start_i{}n{}"
    TT8_before_end  = "TT8_before_end_i{}n{}"
    for i in index_nodes_ids:
        for n in index_person_ids:
            md.addConstr((w_vars[i,n] >= input_global["START"]),    name = TT8_after_start.format(i,n))
            md.addConstr((w_vars[i,n] <= input_global["END"]),      name = TT8_before_end.format(i,n))
    
    
    
    #Bus Travel Constraints
    #These two constants state that person must finish their task and waiting period at node i, 
    #x seconds (controlled by the bus relaxation constant) before the exact bus they want to catch arrives at 
    #BTC1before_dri
    #BTC1after_dri
    constr_BTC1before_dri_name_string   =  "BTC1before_l{}i{}d{}n{}"
    constr_BTC1after_dri_name_string    =  "BTC1after_l{}i{}d{}n{}"
    for l in index_bus_lines_ids:
        for i in route_lnum[l-1]:
            departure_qty = max([key[2] for key in DTime.keys() if (key[0] == l and key[1] == i)])
            for d in range(0, departure_qty):
                for n in index_person_ids:
                    temp_expression_a = w_vars[i, n] + aw_vars[i, n] + bw_vars[i, n]
                    temp_expression_b = 0
                    for t in index_subset_tasks_in[i,n]:
                        temp_expression_b = temp_expression_b + (const_s_t[t] * y_vars[i, t, n]) + (const_st_istar[t] * ts_istar_vars[t])
                    temp_expression_c = DTime[l,i,d]
                    temp_expression_M = M_time * (1 - bus_catch_vars[l, i, d, n])
                    
                    md.addConstr((temp_expression_a + temp_expression_b <= temp_expression_c + temp_expression_M),                  name = constr_BTC1before_dri_name_string.format(l,i,d,n))
                    md.addConstr((temp_expression_a + temp_expression_b >= temp_expression_c - temp_expression_M - bus_relaxation), name = constr_BTC1after_dri_name_string.format(l,i,d,n))
    del temp_expression_a, temp_expression_b, temp_expression_c
    
    #A person can only leave a node once
    #BTC2_lin
    #Also:
    #The person n can only get bus (i, d, r) the relevant Bus_idrn must be positive
    #BTC3_lin
    constr_BTC2_lin_name_string   =  "BTC2_l{}i{}n{}"
    constr_BTC3_lin_name_string   =  "BTC3_l{}i{}n{}"
    for l in index_bus_lines_ids:
        for i in route_lnum[l-1]:
            for n in index_person_ids:
                md.addConstr((bus_catch_vars.sum(l, i, "*", n) <= 1),                       name = constr_BTC2_lin_name_string.format(l,i,n))
                md.addConstr((x_vars.sum(i,"*","BUS",n) <= bus_catch_vars.sum(l,i,"*",n)),  name = constr_BTC3_lin_name_string.format(l,i,n))
    
    #Every time a person gets on a bus from another mode of transport, they must purchase a bus fare 
    #BTC4_lin
    constr_BTC4_ln_name_string   =  "BTC4_l{}n{}"
    for i in index_bus_stops_ids:
        for n in index_person_ids:
            md.addConstr((x_vars.sum("*",i,"BUS",n) + fee_bus_vars[i,n] >= x_vars.sum(i,"*","BUS",n)), name = constr_BTC4_ln_name_string.format(i,n))
                
    #A maximum number of people who can be on a bus at the same time is Nℓ
    #BTC5_lid
    constr_BTC5_lid_name_string    =  "BTC5_l{}i{}d{}"
    for l in index_bus_lines_ids:
        for i in route_lnum[l-1]:
            departure_qty = max([key[2] for key in DTime.keys() if (key[0] == l and key[1] == i)])
            for d in range(0, departure_qty):
                md.addConstr((bus_catch_vars.sum(l, i, d, "*") <= input_objects["BUS_LINES"][l]["MAX_NB_PEOPLE"]), name = constr_BTC5_lid_name_string.format(l,i,d))
    
    
    
    """Bike Constraints"""
    #A person must start and end their bike travel at a bike stop
    #If a person arrives at a non-bike stop via bike, they must leave by bike 
    constr_BT1_jn_name_string    =  "BT1_j{}n{}"
    for j in index_nodes_ids:
        if not j in index_bike_stations_ids:
            for n in index_person_ids:
                    md.addConstr((x_vars.sum("*",j,"CYCLING",n) == x_vars.sum(j,"*","CYCLING",n)), name = constr_BT1_jn_name_string.format(j,n))
    
    
    
    """Personal Spend Constraints"""
    constr_PSC_n_name_string    =  "PSC_n{}"
    for n in index_person_ids:
        temp_expression_a = 0
        for (t,i) in index_personal_tasks[n]:
            temp_expression_a += const_c_t[t] * y_vars[i, t, n]
        temp_expression_b = 0
        for i in index_bus_stops_ids:
            temp_expression_b += fee_bus_vars[i,n] * input_global["COST_BUS_PER_RIDE"]
        temp_expression_c = 0
        for i in index_nodes_ids:
            for j in index_nodes_ids:
                if i != j:
                    temp_expression_c += x_vars[i,j,"CYCLING",n] * const_t_ijm[i, j, "CYCLING"]
        budget_n = input_objects["PEOPLE"][n]["BUDGET"]
        md.addConstr((temp_expression_a + temp_expression_b + temp_expression_c <= budget_n), name = constr_PSC_n_name_string.format(n))
        
    del temp_expression_a, temp_expression_b, temp_expression_c, budget_n
    
    """Optional Test Constraints"""
    #This is an optional constraint to test if the bus functionality works
    constr_OTC_bus_name_string    =  "OTC_bus"
    if force_1_to_catch_a_bus == True:
        md.addConstr((x_vars.sum("*","*","BUS",1) >= 1), name = constr_OTC_bus_name_string)
    
    
    
    md.update()
    md.write(explicit_output_folder_location + "model_export.lp")

    
    """Set objective"""
    objv_time_travelled = 0
    objv_fitness_weighting = 0
    for n in index_person_ids:
        for i,j,m in input_links["NODE_TRAVEL_INFO"].keys():
            objv_time_travelled     += x_vars[i,j,m,n] * const_t_ijm[i,j,m]
            objv_fitness_weighting  += fitness_weighting * x_vars[i,j,m,n] * const_fitness_ijm[i,j,m]
            if m != "BUS":
                objv_time_travelled     += x_vars[j,i,m,n] * const_t_ijm[j,i,m]
                objv_fitness_weighting  += fitness_weighting * x_vars[j,i,m,n] * const_fitness_ijm[j,i,m]
    
    objv_unfinished_task_penality = 0
    for task in input_objects["TASKS"].values():
        t = task["TASK_ID"]
        i = task["PLACE_ID"]
        n = task["PERSON_ID"]
        objv_unfinished_task_penality = objv_unfinished_task_penality + (1 - y_vars[i,t,n]) * unfinished_task_penalty
    
    md.setObjective(objv_time_travelled + objv_fitness_weighting + objv_unfinished_task_penality, GRB.MINIMIZE)
            

    """Compilation of model for export (export is used for model interrogation)"""
    md.update()   
    md.write(explicit_output_folder_location + "model_export.lp")
    
    print("Ready to optimise at: " + str(datetime.now()))
    
    
    """Model Running"""
    #md._vars = vars
    
    #for m in index_modes_of_transport:
    #    for n in [3]:
    #        m._x_vars  = {key : value for key, value in zip(x_vars.keys(), x_vars.values()) if key[3] == 3}
    
    
    #md.optimize()
    
    #md.computeIIS()
    #md.write(explicit_output_folder_location +"infes_model.ilp")
    
    md._x_vars              = x_vars
    md._w_vars              = w_vars
    md._index_nodes_ids     = index_nodes_ids
    md._input_objects       = input_objects  #actions, this can be slimmed down
    md._index_person_ids    = index_person_ids
    md._y_vars              = y_vars
    md._aw_vars             = aw_vars
    
    md.Params.LazyConstraints = 1
    md.setParam('TimeLimit', rum_lim_minutes*60)
    md.update()
    md.optimize(subtourelim)
    print("Complete at: " + str(datetime.now()))

    md._x_vars              = x_vars
    md._w_vars              = w_vars
    
    md._index_nodes_ids     = index_nodes_ids
    md._input_objects       = input_objects
    md._y_vars              = y_vars
    md._aw_vars             = aw_vars
    
    md._index_person_ids    = index_person_ids
    md._y_vars              = y_vars
    md._const_t_ijm         = const_t_ijm
    md._fitness_weighting   = fitness_weighting
    md._const_fitness_ijm   = const_fitness_ijm
    md._unfinished_task_penalty = unfinished_task_penalty
    
    
    #vals = md.getAttr('X', vars)
    #tour = subtour(vals)
    #assert len(tour) == n

    print('')
    #print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g' % md.ObjVal)
    print('')
    
    
    export_results(md, input_global, input_objects, input_links, index_person_ids, index_nodes_ids, scenario_name = scenario_name, show_fig = show_fig)
    
    
    



def export_results(model, input_global, input_objects, input_links, index_person_ids, index_nodes_ids, scenario_name, show_fig):
    
    output_people_routes = dict()
    output_people_route_methods = dict()
    output_people_route_times = dict()
    #ss = " - " #Stands for string spacer
    x_vars = model._x_vars
    w_vars = model._w_vars
    y_vars = model._y_vars
    
    #define tours    
    for n_ in index_person_ids:
        node            = input_objects["PEOPLE"][n_]["HOME_ID"]
        entry_method    = "Home"
        entry_time      = input_global["START"]
        #output_people_routes = [node, entry_method, entry_time]
        edges               = [(i, j, m) for i,j,m,n in x_vars.keys() if x_vars[i,j,m,n].X > 0.5 and n == n_ ]
        tour                = [input_objects["PEOPLE"][n_]["HOME_ID"]]
        tour_methods_entry  = []
        while edges:
            target_link = [(i,j,m) for (i,j,m) in edges if i == tour[-1]]
            if len(target_link) >= 2:
                print("Error - 1")
            if len(target_link) == 0:
                print("Error Detected - Subtour")
                edges = []
                break
            tour                = tour               + [target_link[0][1]]
            tour_methods_entry  = tour_methods_entry + [target_link[0][2]]
            edges.remove(target_link[0])
        output_people_routes[n_] = tour
        output_people_route_methods[n_] = tour_methods_entry
        
    """#define travel times
    output_people_route_methods = dict()
    for n_ in index_person_ids:
        tour_entry_methods = ["Home"]
        for stop in output_people_routes[n_][1:]:
            tour_entry_methods = tour_entry_methods + [w_vars[tour[stop],n_]]"""
        
        
    #define arrival times
    
    for n_ in index_person_ids:
        #entry_time = input_global["START"]
        tour = output_people_routes[n_]
        tour_times = [input_global["START"]]
        for stop in tour[1:]:
            tour_times = tour_times + [w_vars[stop,n_].X]
        output_people_route_times[n_] = tour_times
    #save_values(input_objects, input_links, input_global, output_people_routes, output_people_route_methods, output_people_route_times, model)
    visualise_results_and_export(input_objects, input_links, input_global, output_people_routes, output_people_route_methods, output_people_route_times, scenario_name, model, show_fig)
    


def visualise_results_and_export(input_objects, input_links, input_global, output_people_routes, output_people_route_methods, output_people_route_times, scenario_name, model, show_fig):
    
    object_names_list       = ["PEOPLE", "PLACES", "BIKE_STATIONS", "BUS_STOPS"]
    location_prefix_list    = ["HOME",  "PLACE", "BIKE_STATION", "BUS_STOP"]
    places_x_coords = dict()
    places_y_coords = dict()
    if len(sys.argv) == 1:
        explicit_output_folder_location = str(pathlib.Path(__file__).parent.resolve()) + "\\outputs\\"
    else:
        explicit_output_folder_location = str(pathlib.Path(sys.argv[0]).parent) + "\\outputs\\"
                                        
    
    
    
    for object_name, prefix in zip(object_names_list, location_prefix_list):
        lon_string = prefix + "_LON"
        lat_string = prefix + "_LAT"
        id_string  = prefix + "_ID"
        class_ = input_objects[object_name]
        for object_ in class_.values():
            id                  = object_[id_string]
            places_x_coords[id] = object_[lon_string]
            places_y_coords[id] = object_[lat_string]
    
    max_id  = max(places_x_coords.keys())
    person_modes    = ["WALKING", "CYCLING", "BUS"]

    visited_places_x_coords = dict()
    visited_places_y_coords = dict()
    for id_ in output_people_routes.keys():
        output_people_routes_single = output_people_routes[id_]
        visited_places_x_coords_single = []
        visited_places_y_coords_single = []
        for visted_location in output_people_routes_single:
            visited_places_x_coords_single = visited_places_x_coords_single + [places_x_coords[visted_location]]
            visited_places_y_coords_single = visited_places_y_coords_single + [places_y_coords[visted_location]]
        visited_places_x_coords[id_] = visited_places_x_coords_single
        visited_places_y_coords[id_] = visited_places_y_coords_single
        
    fig_cols    = 3
    fig_rows    = math.ceil(max(output_people_routes.keys()) / fig_cols)
    fig_width   = fig_cols * 6
    fig_height  = fig_rows * 6
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(fig_width, fig_height), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)

    axs = axs.ravel()

    for person_id in output_people_routes.keys():
        fig_id = person_id - 1
        visited_places_x_coords_single = visited_places_x_coords[person_id]
        visited_places_y_coords_single = visited_places_y_coords[person_id]
        
        for stop_number in range(0, len(output_people_route_methods[person_id])):
            label = output_people_route_methods[person_id][stop_number]
            if label == "WALKING":
                color = "green"
            if label == "CYCLING":
                color = "blue"
            if label == "BUS":
                color = "red"
            if label == "Home":
                break
            axs[fig_id].plot([visited_places_x_coords_single[stop_number], visited_places_x_coords_single[stop_number+1]], [visited_places_y_coords_single[stop_number], visited_places_y_coords_single[stop_number+1]], 'ro-', label = label,color = color)

        axs[fig_id].set_xlabel('Long')
        axs[fig_id].set_ylabel('Lat')
        axs[fig_id].set_title('Person ID' + str(person_id))
        axs[fig_id].scatter(list(places_x_coords.values()), list(places_y_coords.values()), s=20)
        axs[fig_id].scatter(list(visited_places_x_coords[person_id]), list(visited_places_y_coords[person_id]), s=20)
        for place_id in places_x_coords.keys():
            axs[fig_id].annotate(place_id, (places_x_coords[place_id], places_y_coords[place_id]))
        #ax1.scatter(range(study_range), pred_input[:study_range], s=20)
        #axs[fig_id].legend(('WALKING',"CYCLING","BUS"))
        
    fig.suptitle(scenario_name[:-4])
    
    plt.savefig(explicit_output_folder_location + scenario_name[:-4] + '.png')
    if show_fig == True:
        plt.show()
    print_result(scenario_name, output_people_routes, output_people_route_methods, output_people_route_times, model)

 
    
def return_component_scores(model):
    index_person_ids        = model._index_person_ids
    x_vars                  = model._x_vars
    y_vars                  = model._y_vars
    const_t_ijm             = model._const_t_ijm
    fitness_weighting       = model._fitness_weighting
    const_fitness_ijm       = model._const_fitness_ijm
    unfinished_task_penalty = model._unfinished_task_penalty
    objv_unfinished_task_penality = 0; objv_time_travelled = 0; objv_fitness_weighting = 0
    for n in index_person_ids:
        for i,j,m in input_links["NODE_TRAVEL_INFO"].keys():
            objv_time_travelled     += x_vars[i,j,m,n].X * const_t_ijm[i,j,m]
            objv_fitness_weighting  += fitness_weighting * x_vars[i,j,m,n].X * const_fitness_ijm[i,j,m]
            if m != "BUS":
                objv_time_travelled     += x_vars[j,i,m,n].X * const_t_ijm[j,i,m]
                objv_fitness_weighting  += fitness_weighting * x_vars[j,i,m,n].X * const_fitness_ijm[j,i,m]
    
    objv_unfinished_task_penality = 0
    for task in input_objects["TASKS"].values():
        t = task["TASK_ID"]; i = task["PLACE_ID"]; n = task["PERSON_ID"]
        objv_unfinished_task_penality = objv_unfinished_task_penality + (1 - y_vars[i,t,n].X) * unfinished_task_penalty
    
    return objv_time_travelled, objv_fitness_weighting, objv_unfinished_task_penality
    
    
def print_result(scenario_name, output_people_routes, output_people_route_methods, output_people_route_times, model):
    if len(sys.argv) == 1:
        explicit_output_folder_location = str(pathlib.Path(__file__).parent.resolve()) + "\\outputs\\"
    else:
        explicit_output_folder_location = str(pathlib.Path(sys.argv[0]).parent) + "\\outputs\\"
    w_vars        = model._w_vars
    y_vars        = model._y_vars
    input_objects = model._input_objects
    aw_vars       = model._aw_vars
    
    with open(explicit_output_folder_location + scenario_name[:-4] + '_output.txt', 'w') as f:
        """Scores"""
        f.write('Scores')
        f.write('Final Score: %g' % model.ObjVal)
        f.write('\n')
        objv_time_travelled, objv_fitness_weighting, objv_unfinished_task_penality = return_component_scores(model)
        f.write('time travelled: '              + str(math.floor(objv_time_travelled*100)/100)          + '\n')
        f.write('fitness score: '               + str(math.floor(objv_fitness_weighting*100)/100)       + '\n')
        f.write('unfinished task penality: '    + str(math.floor(objv_unfinished_task_penality*100)/100)+ '\n')
        f.write('\n')
        
        """Person Routes"""
        f.write('PEOPLE ROUTE')
        f.write('\n')
        for person_id in output_people_routes.keys():
            f.write('PERSON_ID ' + str(person_id) + '\n')
            f.write('NODE_ID  MODE_OF_DEPARTURE TIME_OF_ARRIVAL')
            for node, method, time in zip(output_people_routes[person_id], output_people_route_methods[person_id], output_people_route_times[person_id]):
                time_string = convert_mins_to_time(float(time)) + " (" + str(math.floor(time * 10)/10) + ")"
                f.write(str(node) + " - " + str(method) + " - " + time_string)
                f.write('\n')
            f.write('\n')
    
        f.write('\n')
        """Tasks completed report"""
        f.write('TASKS')
        f.write('\n')
        f.write('ID IS_COMPLETED ARRIVED_TO_NODE STARTED_TIME PERSON_ID LOCATION_ID')
        f.write('\n')
        tasks = input_objects["TASKS"]
        s = " "
        for task_id in tasks.keys():
            PERSON_ID       = tasks[task_id]["PERSON_ID"]
            LOCATION_ID     = tasks[task_id]["PLACE_ID"]
            STARTED_TIME    = w_vars[LOCATION_ID, PERSON_ID].X + aw_vars[LOCATION_ID, PERSON_ID].X
            STARTED_TIME    = convert_mins_to_time(STARTED_TIME)
            ARRIVED_TO_NODE = w_vars[LOCATION_ID, PERSON_ID].X
            ARRIVED_TO_NODE = convert_mins_to_time(ARRIVED_TO_NODE)
            IS_COMPLETED    = str(math.floor(y_vars[LOCATION_ID, task_id, PERSON_ID].X))
            
            
            
            f.write(str(task_id) +s+ IS_COMPLETED +s+ ARRIVED_TO_NODE +s+ STARTED_TIME +s+ str(PERSON_ID) +s+ str(LOCATION_ID))
            f.write('\n')
    print("")
            
    
    

def convert_mins_to_time(input_minutes):
    hours   = math.floor(input_minutes / 60)
    minutes = math.floor(input_minutes - 60 * hours)
    if hours >= 10:
        output_a = str(hours)
    else:
        output_a = "0" + str(hours)
    if minutes >= 10:
        output_b = str(minutes)
    else:
        output_b = "0" + str(minutes)
    return output_a + ":" + output_b

if len(sys.argv) == 1:
    explicit_input_folder_location  = str(pathlib.Path(__file__).parent.resolve()) + "\\inputs\\"
    explicit_output_folder_location = str(pathlib.Path(__file__).parent.resolve()) + "\\outputs\\"
else:
    explicit_input_folder_location  = str(pathlib.Path(sys.argv[0]).parent) + "\\inputs\\"
    explicit_output_folder_location = str(pathlib.Path(sys.argv[0]).parent) + "\\outputs\\"
input_file_names = [f for f in listdir(explicit_input_folder_location) if isfile(join(explicit_input_folder_location, f))]

#this will loop all the inputs we have put in the folder
count = 0
for scenario_input_file in input_file_names:
    input_objects, input_links, input_global = import_inputs(explicit_input_folder_location + scenario_input_file)
    count += 1
    print(scenario_input_file + " start at: " + str(datetime.now()) + str(count) + " / " + str(len(input_file_names)))
    run_scenario(input_objects, input_links, input_global,
                 scenario_name = scenario_input_file, 
                 rum_lim_minutes = rum_lim_minutes, 
                 disable_costly_constraints = False, 
                 force_1_to_catch_a_bus = False,
                 show_fig = False)