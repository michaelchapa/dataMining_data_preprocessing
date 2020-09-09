import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats

########################### bin_Means #####################################
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
        
    print("The respective bin for each value of attribute: \n", set(binValues), "\n")
    print("Computed bins: \n", binEdges, "\n")
    
    binnedValues = pd.DataFrame( \
                   list(zip(data, binValues)), columns = ['value', 'bin'])
    binnedValuesMean = binnedValues.groupby(['bin']).mean()
    print("Mean value for each value in bin: \n", binnedValuesMean, "\n\n")


######################## bin_Boundaries ##################################
# Purpose: 
#   Smooths data by Binning, Prints each bin's closest boundary value.
# Parameters:
#   data  - DataFrame - to be smoothed
#   depth - Integer   - the number of equal length bins
# Returns: 
#   None
# Notes: 
#   Changing pd.cut(labels = False) gives cleaner output
def bin_Boundaries(data, depth):
    data = data['E']
    data = data.sort_values()
    
    binValues, binEdges = pd.cut(data.array, bins = depth, \
                                 labels = range(1, depth + 1), retbins = True)
    
    print("The respective bin for each value of attribute: \n", set(binValues), "\n")
    print("The computed specified bins: \n", binEdges, "\n")
    
    binnedValues = pd.DataFrame( \
                   list(zip(data, binValues)), columns = ['value', 'bin'])
    
    for index, observation in binnedValues.iterrows():
        value = observation[0].tolist()
        minDistance = 999999999
        leastDistant = 0
        
        for edge in binEdges:
            edge = edge.tolist()
            distance = abs(edge - value)
            if distance < minDistance:
                leastDistant = edge
                minDistance = distance
                
        # set value at dataframe
        binnedValues.at[index, 'value'] = leastDistant
        
    print(binnedValues)
       
        
######################## bin_Medians ######################################
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
        
    print("The respective bin for each value of attribute: \n", set(binValues), "\n")
    print("The computed specified bins: \n", binEdges, "\n")
    
    binnedValues = pd.DataFrame( \
                   list(zip(data, binValues)), columns = ['value', 'bin'])
    binnedValuesMedian = binnedValues.groupby(['bin']).median()
    print("Median value for each value in bin: \n", binnedValuesMedian, "\n\n")


######################## pcaAnalysis #####################################
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


####################### calculate_correlation ##############################
# Purpose:
#   Calculates covariance and correlation-coefficient
# Parameters:
#   data - DataFrame - data to be correlated feature-wise
# Returns:
#   None
# Notes:
#   None
def calculate_correlation(data):
    correlations, pValues = stats.spearmanr(data)
    covariances = np.cov(data, rowvar = False)
    labels = ['C', 'D', 'E', 'F']
    
    print('Correlation Coefficient matrix:')
    print(pd.DataFrame(correlations, columns = labels, index = labels), '\n')
    
    print('Covariance matrix: ')
    print(pd.DataFrame(covariances, columns = labels, index = labels), '\n')


#################### construct_contingency_table ##########################        
# Purpose:
#   Constructs a contingency table and Print Chi-square test of independence
#   of variables in the contingency table. 
# Parameters:
#   data - DataFrame - Columns used for table creation
# Returns:
#   None
# Notes:
#   None
def construct_contingency_table(data):
    contingency = pd.crosstab(data['A'], data['B'], normalize = True) 
    c, p, dof, expected = stats.chi2_contingency(contingency)
    aLabels, bLabels = ['a1', 'a2', 'a3'], ['b1', 'b2']

    print('Contingency Table of Normalized Observed Frequencies:\n')
    print(pd.DataFrame( \
          contingency, columns = bLabels, index = aLabels), '\n\n')
    
    print('Expected frequencies, based on marginal sums of the table: \n')
    print(pd.DataFrame(expected, columns = bLabels, index = aLabels), '\n')
    print('Chi-square test statistic: %.4lf' % (c))
    print('P-value of test: %.4lf' % (p))
    print('Degrees of Freedom: %d' % (dof))

    
########################### Main ########################################
data = pd.read_csv('https://raw.githubusercontent.com/michaelchapa' \
                   '/dataMining_data_preprocessing/master/hwk01.csv')
numericalFeatures = data[data.columns[3:]] # remove redundant index column
nominalFeatures = data[data.columns[1:3]]

bin_Means(numericalFeatures, 4) # k = 4, 10, 50
bin_Boundaries(numericalFeatures, 4)
bin_Medians(numericalFeatures, 4)
pcaAnalysis(numericalFeatures, ['C', 'D', 'E', 'F'], 2)
calculate_correlation(numericalFeatures)
construct_contingency_table(nominalFeatures)
