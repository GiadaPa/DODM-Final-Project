"""

@author: Leonora Frangu, Fabio James Greenwood, Giada Palma, Benedetta Pasqualetto

"""

### Libraries
import gurobipy as gp
from gurobipy import *
import pandas as pd
import numpy as np



### Sets and parameters

## variables beginning in "input_request" denote the type of expected inputs, this is fed through methods XXX to dynamically read the input file
input_request_object_classes = {
    "PERSON_ID" : {
        "NAME" : "PEOPLE",
        "HOME_ID" : "Int",
        "HOME_LAT" : "Float",
        "HOME_LON" : "Float",
        "BUDGET" : "INT",
        "MAX_NB_CHANGES_TRANSPORT" : "INT"},

    "PLACE_ID" : {
        "NAME" : "PLACES",
        "PLACE_LAT" : "Float",
        "PLACE_LON" : "Float",
        "MAX_NB_PEOPLE" : "Int"},
    
    "TASK_ID" : {
        "NAME" : "TASKS",
        "PERSON_ID" : "Int",
        "PLACE_ID" : "Int",
        "COST" : "Int",
        "SERVICE_TIME" : "Int",
        "START_TIME" : "Int",
        "END_TIME" : "Int",
        "IS_SPECIAL" : "Int",
        "EXTRA_SERVICE_TIME" : "Int",
        "PENALTY" : "Int"},
    
    "BIKE_STATION_ID" : {
        "NAME" : "BIKE_STATIONS",
        "BIKE_STATION_LAT" : "Float",
        "BIKE_STATION_LON" : "Float",
        "NB_AVAILABLE_BIKES" : "Int",
        "NB_FREE_SPOTS" : "Int"},
    
    "LINE_ID" : {
        "NAME" : "BUS_LINES",
        "START_TIME" : "Int",
        "FREQUENCY" : "Int",
        "MAX_NB_PEOPLE" : "Int"},
    
    "BUS_STOP_ID" : {
        "NAME" : "BUS_STOPS",
        "BUS_STOP_LAT" : "Float",
        "BUS_STOP_LON" : "Float"}
    }

input_request_links = {
    "BUS_STOP_ID" : {
        "NAME" : "BUS_STOP_TO_LINE",
        "BUS_LINE_ID" : "Int"},
    
    "NODE_ID" : {
        "NAME" : "NODE_TRAVEL_INFO",
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




# Function read all information in the input file
# Method reads first line of each paragraph to understand which class the input belongs to, 
# then uses/references the input_request dictionaries to understand the expect data and format within that paragraph
def horizon():
    input_request_names = list(input_request_object_classes.keys()) + list(input_request_links.keys()) + list(input_request_global.keys())
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
                    
                    input_class = input_request_object_classes[lines[current_line].split()[0]]
                    ID = int(lines[current_line].split()[1])
                    #if not input_class["NAME"] in input_object_classes.keys():
                    #    input_object_classes[input_class["NAME"]] = {}
                    
                    #READ ALL INFO ON CLASS INSTANCE
                    reading_class_instance = True
                    current_line += 1
                    while reading_class_instance == True:
                        
                        #check for end of class instance
                        if lines[current_line] == "\n":
                            current_line += 1
                            break
                        
                        if not lines[current_line].split()[0] in input_object_classes.keys():
                            input_object_classes[lines[current_line].split()[0]] = {}
                        
                        match input_class[lines[current_line].split()[0]]:
                            case "Int":
                                input_var = int(lines[current_line].split()[1])
                            case "ID":
                                input_var = int(lines[current_line].split()[1])
                            case "Float":
                                input_var = float(lines[current_line].split()[1])
                            case "String":
                                input_var = lines[current_line].split()[1]
                            case _:
                                raise Exception("Error: format " + input_class[lines[current_line].split()[0]] + " not found")

                        input_object_classes[lines[current_line].split()[0]][ID] = input_var
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
                    
                
                    
                        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
            
            
            
            
                
                    
            
            
            
            
        
        #data_inputs.update({'start': to_hours(line.split()[1]), 'end': to_hours(line.split()[2])})
        print(data_inputs)

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