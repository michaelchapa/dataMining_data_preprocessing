import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Purpose: 
#   Smooths data by Binning, Prints each bin's mean value.
# Parameters:
#   data  - DataFrame - to be smoothed
#   depth - Integer   - the number of equal length bins
# Returns: 
#   None
# Notes: 
#   Changing pd.cut(labels = False) gives cleaner output
def bin_Means(data, depth):
    data = data['F'] # Creates Series
    data = data.sort_values()
    
    binValues, binEdges = pd.cut(data.array, bins = depth, \
                labels = range(1, depth + 1), retbins = True)
        
    print("The respective bin for each value of attribute: \n", binValues, "\n")
    print("The computed specified bins: \n", binEdges)
    
    binnedValues = pd.DataFrame( \
                   list(zip(data, binValues)), columns = ['value', 'bin'])
    binnedValuesMean = binnedValues.groupby(['bin']).mean()
    print("Means for each bin: \n", binnedValuesMean)
    
# Purpose: 
#   Smooths data by Binning, Prints each bin's median value.
# Parameters:
#   data  - DataFrame - to be smoothed
#   depth - Integer   - the number of equal length bins
# Returns: 
#   None
# Notes: 
#   Pretty much the same as bin_Means function
def bin_Medians(data, depth):
    data = data['F'] # Creates Series
    data = data.sort_values()
    
    binValues, binEdges = pd.cut(data.array, bins = depth, \
                labels = range(1, depth + 1), retbins = True)
        
    print("The respective bin for each value of attribute: \n", binValues, "\n")
    print("The computed specified bins: \n", binEdges)
    
    binnedValues = pd.DataFrame( \
                   list(zip(data, binValues)), columns = ['value', 'bin'])
    binnedValuesMedian = binnedValues.groupby(['bin']).median()
    print("Medians for each bin: \n", binnedValuesMedian)
    
# Purpose:
#   Uses Primary Component Analysis to decompose (reduce) data to 'p' columns.
# Parameters:
#   data        - DataFrame - to be decomposed
#   columnNames - List      - the data's column names
#   p           - Integer   - Reduce columns into p new columns
# Returns:
#   None
# Notes:
#   on function call: p value must not exceed number of features (columnNames)
def pcaAnalysis(data, columnNames, p):
    data = data[columnNames]
    features = data.to_numpy()
    
    pca = PCA(n_components = p)
    pca.fit(features)
    
    print('The amount of variance explained by each of the selected components: ')
    print(pca.explained_variance_, '\n\n')
    
    print('The values corresponding to each of the selected components: ')
    print(pca.singular_values_, '\n')
    print('The values are equal to the 2-norms of the %d variables' \
          ' in the lower-dimensional space.\n\n' % (p))
        
    print('Per Feature empirical mean, estimated from the training set:')
    print(pca.mean_)

# Purpose:
#
# Parameters:
#
# Returns:
#
# Notes:
# 

# Purpose:
#
# Parameters:
#
# Returns:
#
# Notes:
# 
########################### MAIN ########################################
data = pd.read_csv('https://raw.githubusercontent.com/michaelchapa' \
                   '/dataMining_data_preprocessing/master/hwk01.csv')
numericalFeatures = data[data.columns[3:]] # remove redundant index column


bin_Means(numericalFeatures, 4) # k = 4, 10, 50
bin_Medians(numericalFeatures, 4)
# pcaAnalysis(numericalFeatures, ['C', 'D', 'E', 'F'], 2)
