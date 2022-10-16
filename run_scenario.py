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

    disable_costly_constraints = True
    """Begin of model creation """
    md = gp.Model()


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
    #n -> person
    index_person_ids = input_objects["PEOPLE"].keys()    
    
    #t -> task number
    
    #p -> place 

    #b -> bike station

    #l -> bus line number 
    index_bus_lines = input_objects["BUS_LINES"].keys()

    #d -> departure time number d.Each bus stops have a number of departure times,each of which are indexed with a number
    #index_bus_departures = dict()
    #df_bus_stops = pd.DataFrame(input_links["BUS_STOP_TO_LINE"].values(), index=input_links["BUS_STOP_TO_LINE"].keys())
    #depots = df_bus_stops[df_bus_stops[0]==1]
    
    
    """for l in index_bus_lines:
        #       {k  : v   for k,   v     in points.items()                                                                   if v[0] < 5    and v[1] < 5}
        depot_id = {key: value for key, value in zip(input_links["BUS_STOP_TO_LINE"].keys(), input_links["BUS_STOP_TO_LINE"].values()) if int(key[1]) == l and value == 1}
        depot_id = depot_id[0]
        temp_index_stops_id = {key[0]: value for key, value in zip(input_links["BUS_STOP_TO_LINE"].keys(), input_links["BUS_STOP_TO_LINE"].values()) if int(key[1]) == l}
        for link in range(0, len(temp_index_stops_id)-1):
            print("F")"""
        
        
    
    
    #bikePeriod → period time for the bike quantity/spaces relaxation/assumption

    #m -> transportation (m)ode
    index_modes_of_transport = ["WALKING", "CYCLING", "BUS"]

    
    
    
    
    
    
    
    #r          -> bus route number r
        
    
    #t          -> task number t
    index_task_ids = input_objects["TASKS"].keys()
    #bikePeriod -> period time for the bike quantity spaces relaxation assumption

    """Special Subsets"""
    #Tasks related to node i and person n
    index_subset_tasks_in = dict()
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            index_subset_tasks_in[node_id, person_id] = []
           
    for task_id in input_objects["TASKS"].keys():
        node_id     = input_objects["TASKS"][task_id]["PLACE_ID"]
        person_id   = input_objects["TASKS"][task_id]["PERSON_ID"]
        task_id     = input_objects["TASKS"][task_id]["TASK_ID"]
        index_subset_tasks_in[node_id, person_id] = index_subset_tasks_in[node_id, person_id] + [task_id]

    #node number num along line l starting with depot
    #route[l, num]
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
    
    route_time_delay = []
    for line in route_lnum:
        route_time_delay_single = [0]
        for i in range(0,len(line)-1):
            time_delay = route_time_delay_single[-1] + [values["TIME"] for key, values in zip(NODE_TRAVEL_INFO.keys(), NODE_TRAVEL_INFO.values()) if (int(key[0]) == line[i] and int(key[1]) == line[i+1] and key[2] =="BUS")][0]
            route_time_delay_single = route_time_delay_single + [time_delay]
        route_time_delay = route_time_delay + [route_time_delay_single]
        
    del BUS_STOP_TO_LINE, NODE_TRAVEL_INFO, route_time_delay_single, route_lnum_single, origin, links_shortlist_a, links_shortlist_b, links_shortlist, node           
    print("Hello")    


       


    """Constants"""
    
    #cost of a task

    #task time window
    const_a_t = dict()
    const_b_t = dict()
    
    for task in input_objects["TASKS"].values():
        const_a_t[task["TASK_ID"]] = task["START_TIME"]
        const_b_t[task["TASK_ID"]] = task["END_TIME"]
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

    #set of bus departure times (line l, node i, departure time d)
    start_time  = input_global["START"]
    end_time    = input_global["END"]
    DTime = dict()
    for l in index_bus_lines:
        freq = input_objects["BUS_LINES"][1]["FREQUENCY"]
        for x in range(0,len(route_time_delay[l-1])):
            current_time = start_time
            d = 0
            while current_time < end_time:
                node_id = route_lnum[l-1][x]
                DTime[l,node_id,d] = current_time
                d += 1
                current_time += freq



    #time window a person has to board a bus (see assumptions)
    bus_relaxation = 2 #minutes

    #Maximum number of people allowed on a bus (see assumptions)

    #the time required to travel down arc ij via mode m

    #Penalty for a task t not performed

    #Max number of times a person n can change transportation mode

    #Boolean whether node i requests service from person n

    #Boolean whether node i is person n's home

    #Set of all starting points of person n
    const_h_in = dict()
    for person_id in index_person_ids:
        for node_id in index_nodes_ids:
            const_h_in[node_id, person_id] = 0
    for person_id in index_person_ids:
        node_id = input_objects["PEOPLE"][person_id]["HOME_ID"]
        const_h_in[node_id, person_id] = 1
        
    #travelling time associated to each arc
    const_t_ijm = dict()
    for i_id, j_id, mode_id in input_links["NODE_TRAVEL_INFO"].keys():
        value = input_links["NODE_TRAVEL_INFO"][i_id, j_id, mode_id]["TIME"]
        const_t_ijm[i_id, j_id, mode_id] = value
    del value
    md.update()
    
    #this is the const_t_ijm extrended by an extra dimention (n) for use in a later constraint
    """I will want to delete this constant"""
    const_t_ijmn = dict()
    for i_id, j_id, mode_id in input_links["NODE_TRAVEL_INFO"].keys():
        for person_id in index_person_ids:
            value = input_links["NODE_TRAVEL_INFO"][i_id, j_id, mode_id]["TIME"]
            const_t_ijmn[i_id, j_id, mode_id, person_id] = value
    del value
    md.update()
    
    #whether a task is special (can be delayed)
    const_special_t = dict() #
    for task in input_objects["TASKS"].values():
        const_special_t[task["TASK_ID"]] = task["IS_SPECIAL"]
    md.update() 
    
    
    
    """M (large) Constants"""
    M_time = input_global["END"] * 100
    

    """Independent Variables"""
    
    
    #Amount of money of person n
    
    
    #whether person n travels down arc ij on tranportation mode m (Bool)
    list_to_combine = [list(input_links["NODE_TRAVEL_INFO"].keys()), list(index_person_ids)]
    links_person_combination_list_pre = list(itertools.product(*list_to_combine))
    links_person_combination_list = []
    for instance in links_person_combination_list_pre:
        links_person_combination_list = links_person_combination_list + [instance[0] + (instance[1], )]
    link_person_combined_dict = dict.fromkeys(links_person_combination_list)
      
    x_vars = md.addVars(link_person_combined_dict.keys(), vtype=GRB.BINARY, name='x_var') 
    
    
    #whether person n completes task t at node i
    y_vars = gp.tupledict()
    y_var_string = "y_var_i{}_t{}_n{}"
    for tasks_id in input_objects["TASKS"].keys():
        location_id = input_objects["TASKS"][tasks_id]["PLACE_ID"]
        person_id = input_objects["TASKS"][tasks_id]["PERSON_ID"]
        y_vars[location_id, tasks_id, person_id] = md.addVar(vtype=GRB.BINARY, name=y_var_string.format(location_id, tasks_id, person_id))
    
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
    
    
    #whether fare for bus is incurred for node i,line l,departure time d
    
    
    #Boolean whether node i is serviced by person n
    
    
    #health or loss gain according to transportation mode chosen
    
    
    #waiting idle time at node (pre-task)
    aw_vars = gp.tupledict()
    aw_var_string = "aw_vars_i{}__n{}"
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            aw_vars[node_id, person_id] = md.addVar(vtype=GRB.CONTINUOUS, name=aw_var_string.format(node_id, person_id))
    
        
    #waiting idle time at node (post-task)
    bw_vars = gp.tupledict()
    bw_var_string = "bw_vars_i{}__n{}"
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            bw_vars[node_id, person_id] = md.addVar(vtype=GRB.CONTINUOUS, name=bw_var_string.format(node_id, person_id))
    
    
    
    """Semi-Dependant Variables"""
    #These are the variables that are technically dependant variables but are modelled as constrained independent variables
    print("Semi-Dependant Variables")
    w_vars = gp.tupledict()
    w_var_string = "w_var_i{}_n{}"
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            w_vars[node_id, person_id] = md.addVar(vtype=GRB.CONTINUOUS, name=w_var_string.format(node_id, person_id))

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
    print("BCoFO -> (out)")
    constr_BCoFO_string = "const_BCoFO_np[{},{}]"    
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            md.addConstr((x_vars.sum(node_id, "*", "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id) + const_h_in[node_id, person_id]) >= 0 ), name = constr_BCoFO_string.format(node_id, person_id))
            #md.addConstr((x_vars.sum(node_id, "*", "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id)) >= 0 ), name = "BCoFO") 
    del constr_BCoFO_string
    
    # "BCoFI" -> (In)
    constr_BCoFI_string = "const_BCoFI_np[{},{}]"    
    for node_id in index_nodes_ids:
        for person_id in index_person_ids:
            md.addConstr((x_vars.sum("*", node_id, "*", person_id) - (0.5 * y_vars.sum(node_id, "*", person_id) + const_h_in[node_id, person_id]) >= 0 ), name = constr_BCoFI_string.format(node_id, person_id))
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
    
    
    """Dont DEL!!!! This is how to multiply a matrix of variables by a simular matrix of constrants"""
    """md.addConstr((x_vars.prod(const_t_ijmn, "*", node_id, "*", person_id)  >= 0 ), name = "Test")"""
    
    #Task Timing
    #This controls the time (w) which the task at j starts
    #TT1_ijntm
    print("TT1_ijntm")
    
    constr_TT1_name_string =  "TT1_i{}j{}n{}t{}m{}"
    if disable_costly_constraints == False:
        for i in index_nodes_ids:
            print(i)
            for j in index_nodes_ids:
                for n in index_person_ids:
                    for t in index_task_ids:
                        for m in index_modes_of_transport:
                            if i != j and return_if_valid_reference(x_vars, [i, j, m, n], False, True):
                                expr_a_temp = w_vars[j, n] - w_vars[i, n] - aw_vars[j, n] - bw_vars[i, n]
                                expr_b_temp = - x_vars[i, j, m, n] * const_t_ijm[i,j,m]
                                #These expresions only count if there is a task for the person/node/task combination
                                expr_c_temp = 0
                                for t in index_subset_tasks_in[i,n]:
                                    expr_c_temp = expr_c_temp - (const_s_t[t] * y_vars[i, t, n]) - (const_st_istar[t] * ts_istar_vars[t])
                                    
                                expr_d_temp = - M_time * (1 - x_vars[i,j,m,n])
                                
                                
                                md.addConstr((expr_a_temp + expr_b_temp + expr_c_temp - expr_d_temp >= 0), name = constr_TT1_name_string.format(i,j,n,t,m,))
                                
                                """Do not delete, this was the previous requirement to make the constraint, but is still a good example to kepp"""
                                """expression_temp = 0
                                for j in index_nodes_ids:
                                    for m in index_modes_of_transport:
                                        if return_if_valid_reference(x_vars, [i, j, m, n], False, True):
                                            expression_temp += x_vars[i, j, m, n] * const_t_ijm[i,j,m] 
                                md.addConstr((expression_temp>=1), name = "Test2")"""

                                #md.addConstr((sum(x_vars[i, j, m, n] * const_t_ijm[i,j,m] for j in index_nodes_ids for m in index_modes_of_transport if return_if_valid_reference(x_vars, [i, j, m, n], False, True))>1), name = "Test2")
        del expr_a_temp, expr_b_temp, expr_c_temp, expr_d_temp
    
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
                    md.addConstr((const_a_t[t] - M_time * tstar_vars[t] - M_time * (1 - y_vars[i, t, n]) <= w_vars[i, n]), name = TT2after_name_string.format(i,n,t))
                    md.addConstr((const_b_t[t] + M_time * tstar_vars[t] + M_time * (1 - y_vars[i, t, n]) >= w_vars[i, n]), name = TT2before_name_string.format(i,n,t))
                    #Note: Not sure the below constraint is needed
                    md.addConstr((tstar_vars[tasks_id] <= y_vars[i, t, n]), name = TT3_name_string.format(i,n,t))
    
    #Task extensions/delays only apply to tasks that are special
    #TT4_t
    TT4_name_string =  "TT4_t{}"
    for t in index_task_ids:
        md.addConstr((tstar_vars[tasks_id] <= const_special_t[t]), name = TT4_name_string.format(t))
    
    #A person will only complete a single task at each node
    #TT5_t_n
    TT5_name_string =  "TT5_t{}n{}"
    for i in index_nodes_ids:
        for n in index_person_ids:
            md.addConstr((y_vars.sum(i, "*", n) <= 1), name = TT5_name_string.format(t,n))
        
    
    #Bus Travel Constraints
    #These two constants state that person must finish their task and waiting period at node i, 
    #x seconds (controlled by the bus relaxation constant) before the exact bus they want to catch arrives at 
    #BTC1after_dri
    #BTC1before_dri
    constr_BTC1after_dri_name_string =  "BTC1after_l{}i{}d{}"
    for l in index_bus_lines:
        for i in route_lnum[l]:
            departure_qty = max([value[2] for key in DTime.keys() if (key[0] == l and key[1] == i)])
            for d in range(0, departure_qty + 1):
                for person_id in index_person_ids:
                    temp_expression_a = w_vars[node_id_along_line, person_id] + bw_vars[node_id, person_id]
                    temp_expression_b = 0
                    for t in index_subset_tasks_in[i,n]:
                        temp_expression_b = temp_expression_b - (const_s_t[t] * y_vars[i, t, n]) - (const_st_istar[t] * ts_istar_vars[t])
                    temp_expression_c = DTime[l,i,d] + M_time * (1 - bus_catch_vars[l,i,d]) + bus_relaxation
                    md.addConstr((temp_expression_a + temp_expression_b <= temp_expression_c), name = constr_BTC1after_dri_name_string.format(l,i,d))
                    
                    
                    
                    #+ const_s_t[]
    
    

    
    
    
    
    
    
    
                    
                
    md.update()
    md.write(explicit_output_folder_location + "model_export.lp")
                    
                
        
    
    
    
    
    
    
    
    
    print("Stop")
    
    
    
    
    
    
    
    
    
    
    
    
            
            
    """Compilation of model for export (export is used for model interrogation)"""
    md.update()   
    md.write(explicit_output_folder_location + "model_export.lp")
    
    
    
    
    """Model Running"""

    md._vars = vars
    md.Params.LazyConstraints = 1
    md.optimize(subtourelim)

    vals = md.getAttr('X', vars)
    tour = subtour(vals)
    assert len(tour) == n

    print('')
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g' % md.ObjVal)
    print('')

"""Temporary section"""
#This is a temporary section to allow for the inporting of inputs and running of the model function (above) 
#for a single run for the proposes of development

explicit_input_folder_location = "C:/Users/fabio/OneDrive/Documents/Studies/Discrete_Optimisation/DODM-Final-Project/Demo Instances/inputs_test/"
explicit_output_folder_location = "C:/Users/fabio/OneDrive/Documents/Studies/Discrete_Optimisation/DODM-Final-Project/Demo Instances/outputs_test/"
input_file_names = [f for f in listdir(explicit_input_folder_location) if isfile(join(explicit_input_folder_location, f))]
input_objects, input_links, input_global = import_inputs(explicit_input_folder_location + input_file_names[0])
print("G")


            
            
            
            

run_scenario(input_objects, input_links, input_global)    
    
