import sys
import pandas as pd

from .math import getMean, getStd
from .logs import printLog, printError

def calcMeans(dataframe, features):
    means = {}
    for feature in features:
        new_mean = {feature: getMean(dataframe[feature])}
        means.update(new_mean)
    return means    

def calcStd(dataframe, features):
    stds = {}
    for feature in features:
        new_std = {feature: getStd(dataframe[feature])}
        stds.update(new_std)
    return stds

def normalizeData(dataframe, means=None, stds=None):
    numericalFeatures = dataframe.select_dtypes(include=['float64'])
    studentsData = []
    if means == None:    
        normalizer = Normalizer(dataframe=dataframe, features=numericalFeatures)
    else:
        normalizer = Normalizer(means=means, stds=stds)
    
    for feature in numericalFeatures:
        normalizer.cleanNan(dataframe, feature)

    for i in range(len(dataframe)):
        newData = {'features': {}}
        for feature in numericalFeatures:
            newData['features'][feature] = normalizer.normalize(dataframe[feature][i], feature)
        
        if 'Best Hand' in dataframe.columns:
            bestHand = dataframe['Best Hand'][i]
            newData['features']['Best Hand'] = normalizer.normalizeHand(bestHand)

        if 'Hogwarts House' in dataframe.columns:    
            house = dataframe['Hogwarts House'][i]
            newData['label'] = house

        studentsData.append(newData)

    return normalizer, studentsData


class Normalizer:
    def __init__(self, dataframe=None, features=None, means=None, stds=None):
        if means == None:
            self.means = calcMeans(dataframe, features)
            self.stds = calcStd(dataframe, features)
        else:
            self.means = means
            self.stds = stds

    def normalize(self, value, feature):
        mean = self.means[feature]
        std = self.stds[feature]
        return (value - mean) / std

    def denormalize(self, value, feature):
        mean = self.means[feature]
        std = self.stds[feature]
        return (value * std) + mean

    def normalizeHand(self, hand):
        if hand == "Left":
            return [1, 0]
        elif hand == "Right":
            return [0, 1]

    def denormalizeHand(self, hand):
        if hand == [1, 0]:
            return "Left"
        elif hand == [0, 1]:
            return "Right"

    def cleanNan(self, dataframe, feature):
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

        median = dataframe[feature].median()
        dataframe.loc[dataframe[feature].isna(), feature] = median