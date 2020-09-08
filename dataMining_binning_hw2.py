import pandas as pd
import numpy as np

# Purpose - Smooths data using DataFrame cut()
# Parameters:
#   data - DataFrame - to be smoothed
#   depth - Integer - the number of equal length bins
# Returns: 
#   None
# Comments: Changing pd.cut(labels = False) gives cleaner output
def bin_Means(data, depth):
    data = data['F'] # Creates Series
    data = data.sort_values()
    binValues, bins = pd.cut(data.array, bins = depth, \
                labels = False, retbins = True)
        
    print("The respective bin for each value of attribute: \n", binValues, "\n")
    print("The computed specified bins: \n", bins)
    
def bin_Boundaries(data, depth):
    data = data['F']
    data = data.sort_values()
    binValues, bins = pd.cut
    
########################### MAIN ########################################
data = pd.read_csv('https://raw.githubusercontent.com/michaelchapa' \
                   '/dataMining_data_preprocessing/master/hwk01.csv')
data = data[data.columns[1:]] # remove redundant index column


bin_Means(data, 100) # k = 100