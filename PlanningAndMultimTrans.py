"""

@author: Leonora Frangu, Fabio James Greenwood, Giada Palma, Benedetta Pasqualetto

"""

### Libraries
from cmath import isnan
import gurobipy as gp
from gurobipy import *
import pandas as pd
import numpy as np
import copy


### Sets and parameters

## variables beginning in "input_request" denote the type of expected inputs, this is fed through methods XXX to dynamically read the input file
input_request_object_classes = {
    "PEOPLE" : {
        "PERSON_ID" : "ID",
        "HOME_ID" : "Int",
        "HOME_LAT" : "Float",
        "HOME_LON" : "Float",
        "BUDGET" : "INT",
        "MAX_NB_CHANGES_TRANSPORT" : "INT"},

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
        "NODE_ID" : "Int",
        "MODE" : "String",
        "DISTANCE" : "Float",
        "TIME" : "Float",
        "FITNESS" : "Float"}
    }

input_request_global = {
    "HORIZON" : {
        "START" : "Int",
        "END" : "Int"},
    
    "MODES_OF_TRANSPORTATION" : {
        "ID" : "Name"},
    
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



# Dictionary definition of the horizon
data_inputs = {}
input_object_classes = {}
input_links = {}
input_global = {}

#update_data_input(data_input_table, value, format)
#def update_data_input(data_input_table, value_name, value, format, object_name = "None", object_classes = input_object_classes.keys()):
    
    
    



# Function read all information in the input file
# Method reads first line of each paragraph to understand which class the input belongs to, 
# then uses/references the input_request dictionaries to understand the expect data and format within that paragraph
def horizon():
    input_request_names = list(input_request_object_classes.keys()) + list(input_request_global.keys())
    with txtFile as f:
        lines = f.readlines()
        line_qty = len(lines)
        currently_reading = True
        current_line = 0
        while currently_reading == True:
            
            #terminate loop
            if current_line >= line_qty:
                break
            
            #skip blank lines
            if lines[current_line] == "\n":
                current_line += 1
                continue
                        
            #determine if paragraph is the start of a new date point and act accordingly
            # if input in input_request_links
            if lines[current_line].split()[0] == "BUS_STOP_ID"  and lines[current_line].split()[1] == "BUS_STOP_ID":
                print("Hello")
            elif lines[current_line].split()[0] == "NODE_ID"    and lines[current_line].split()[1] == "NODE_ID":
                    print("Hello")
                    
            #if not in input_request_links
            elif lines[current_line].split()[0] in input_request_names:
                # if input in input_request_object_classes
                if lines[current_line].split()[0] in input_request_object_classes.keys():
                    
                    current_class_name = copy.deepcopy(lines[current_line].split()[0])
                    input_class = input_request_object_classes[lines[current_line].split()[0]]
                    reading_current_class = True
                    current_line += 1
                    
                    #loop for reading all values of class X                    
                    while reading_current_class == True:
                        
                        #detect if the next class has arrived
                        if lines[current_line].split()[0] in input_request_names or (lines[current_line].split()[0] == "BUS_STOP_ID"  and lines[current_line].split()[1] == "BUS_STOP_ID") or (lines[current_line].split()[0] == "NODE_ID"    and lines[current_line].split()[1] == "NODE_ID"):
                            reading_current_class = False
                            continue
                        
                        #detect if there is an ID for class
                        try:
                            if input_class[lines[current_line].split()[0]] == "ID":
                                ID = lines[current_line].split()[1]
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
                            
                            input_object_classes = update_input_table(input_object_classes, input_value=lines[current_line].split()[1], input_name=lines[current_line].split()[0], input_class = input_class, ID = ID)
                            current_line += 1
                        
                # if input in input_request_global
                elif lines[current_line].split()[0] in input_request_global.keys():
                    input_class = input_request_global[lines[current_line].split()[0]]
                    if not isinstance(input_class, dict):
                        input_global[lines[current_line].split()[0]] = lines[current_line].split()[1]
                    else:
                        for key, i in zip(input_class.keys(), range(1, len(input_class.keys())+1)):
                            input_global[key] = lines[current_line].split()[i]
                    current_line += 1    
            
            else:
                current_line += 1
                    
def update_input_table(input_table, input_value=lines[current_line].split()[1], input_name=lines[current_line].split()[0], input_class = np.nan, ID = np.nan):
    
    if not input_name in input_table.keys():
        input_table[input_name] = {}

    match input_class[input_name]:
        case "Int":
            input_var = int(input_value)
        case "ID":
            input_var = int(input_value)
        case "Float":
            input_var = float(input_value)
        case "String":
            input_var = input_value
        case _:
            raise Exception("Error: format " + input_class[input_name] + " not found")

    if pd.isnan(ID):
        input_table[input_name] = input_var
    else:
        input_table[input_name][int(ID)] = input_var
    
    return input_table
                    


horizon() # Check if the horizon is correct

print("Hello")

people = []     #Declare an empty list for the people.

### Initialization of the problem

### Variables

### Constraints

### Objective function

### Resolution
# Calling the solver
# Printing an optimal solution (if it has been found)