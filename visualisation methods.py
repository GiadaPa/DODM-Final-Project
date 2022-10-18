from matplotlib import colors
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from os import listdir
from os.path import isfile, join
from import_function import import_inputs

explicit_input_folder_location = "C:/Users/fabio/OneDrive/Documents/Studies/Discrete_Optimisation/DODM-Final-Project/Demo Instances/inputs_test/"
explicit_output_folder_location = "C:/Users/fabio/OneDrive/Documents/Studies/Discrete_Optimisation/DODM-Final-Project/Demo Instances/outputs_test/"
input_file_names = [f for f in listdir(explicit_input_folder_location) if isfile(join(explicit_input_folder_location, f))]
input_objects, input_links, input_global = import_inputs(explicit_input_folder_location + input_file_names[0])

 
# reading the database
#data = pd.read_csv("tips.csv")
 
places_x_coords = [place["PLACE_LON"] for place in input_objects["PLACES"].values()]
places_y_coords = [place["PLACE_LAT"] for place in input_objects["PLACES"].values()]
person_route    = [1,2,3]
person_modes    = ["WALKING", "BUS"]

visited_places_x_coords = []
visited_places_y_coords = []
for i in person_route:
    visited_places_x_coords = visited_places_x_coords + [places_x_coords[i]]
    visited_places_y_coords = visited_places_y_coords + [places_y_coords[i]]


for i in range(0, len(visited_places_x_coords)-1):
        match person_modes[i]:
            case "WALKING":
                color = "green"
            case "CYCLING":
                color = "orange"
            case "BUS":
                color = "red"
        
        plt.plot([visited_places_x_coords[i], visited_places_x_coords[i+1]], [visited_places_y_coords[i], visited_places_y_coords[i+1]], 'ro-', color = color)
        


plt.xlabel('Long')
plt.ylabel('Lat')
plt.title('Route Map')
plt.scatter(places_x_coords, places_y_coords, s=20)
plt.scatter(visited_places_x_coords, visited_places_y_coords, s=20)
for i in range(0, len(places_x_coords)):
    plt.annotate(i, (places_x_coords[i], places_y_coords[i]))
#ax1.scatter(range(study_range), pred_input[:study_range], s=20)
plt.legend(('Places',"visited"))

 
plt.show()

