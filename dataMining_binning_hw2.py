import pandas as pd
import numpy as np

# Purpose - Smooths data using DataFrame cut() and qcut() functions
# Parameter - DataFrame
# Returns - DataFrame
def binData(data, depth):
    data = data['F']
    array = data.array
    out = pd.cut(array, depth)
    
    print(out)
    
    
########################### MAIN ########################################
data = pd.read_csv('https://raw.githubusercontent.com/michaelchapa' \
                   '/dataMining_data_preprocessing/master/hwk01.csv')
data = data[data.columns[1:]] # remove redundant index column


binData(data, 100) # k = 100