"""

@author: Leonora Frangu, Fabio James Greenwood, Giada Palma, Benedetta Pasqualetto

"""

### Librariesgt
from cmath import isnan
import gurobipy as gp
from gurobipy import *
import pandas as pd
import numpy as np
import copy


### Sets and parameters

## variables beginning in "input_request" denote the type of expected inputs, this is fed through methods XXX to dynamically read the input file
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
data_inputs = {}
input_objects = {}
input_links = {}
input_global = {}

#update_data_input(data_input_table, value, format)
#def update_data_input(data_input_table, value_name, value, format, object_name = "None", objects = input_objects.keys()):
    
    
    



# Function read all information in the input file
# Method reads first line of each paragraph to understand which class the input belongs to, 
# then uses/references the input_request dictionaries to understand the expect data and format within that paragraph
def import_inputs(input_objects = input_objects, input_links = input_links, input_global = input_global):
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
        
    
    match input_format:
        case "Int":
            input_var = int(input_value)
        case "ID":
            input_var = int(input_value)
        case "Float":
            input_var = float(input_value)
        case "String":
            input_var = input_value
        case _:
            raise Exception("Error: format " + input_format + " not found")
            

    if pd.isnull(ID):
        input_table[input_name] = input_var
    else:
        input_table[object_class_name][int(ID)][input_name] = input_var
    
    return input_table
                    

input_data['horizon'] = get_horizon()
#print(input_data)
"""
#----------------------------------------------------------------------------------------------------------------

input_objects, input_links, input_global = import_inputs(input_objects, input_links, input_global)


print("Hello")

people = []     #Declare an empty list for the people.

print(get_horizon())
### Initialization of the problem

### Variables

### Constraints

### Objective function

### Resolution
# Calling the solver
# Printing an optimal solution (if it has been found)