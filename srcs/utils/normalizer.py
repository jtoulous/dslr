import sys
import pandas as pd

from utils.math import getMean, getLength, getStd, getMin, getPercentile, getMax 

#def calcMeans(dataframe, features):
#    means = {}
#    for feature
#
#class Normalizer:
#    def __init__(dataframe, features):
#        self.means = calcMeans(dataframe, features)
#        self.standardDev = {}

        


def normalizeData():
    dataset = pd.read_csv(sys.argv[1])
    dataset = dataset.drop(columns=["Index", "First Name", "Last Name", "Birthday"])

    categoricalFeatures = dataset.select_dtypes(include=['object'])
    print(categoricalFeatures.to_string(index=False))

    return dataset