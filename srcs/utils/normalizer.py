import sys
import pandas as pd

from utils.math import getMean, getStd

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

def normalizeData(dataframe):
    numericalFeatures = dataframe.select_dtypes(include=['float64'])
    normalizer = Normalizer(dataframe, numericalFeatures)
    featuresData = []
    labelsData = []
    
    for feature in numericalFeatures:
        normalizer.cleanNan(dataframe, feature)

    for i in range(len(dataframe)):
        new_data = {}
        for feature in numericalFeatures:
            new_data[feature] = normalizer.normalize(dataframe[feature][i], feature)
        
        if 'Best Hand' in dataframe.columns:
            bestHand = dataframe['Best Hand'][i]
            new_data['Best Hand'] = normalizer.normalizeHand(bestHand)

        house = dataframe['Hogwarts House'][i]
        labelsData.append(normalizer.normalizeHouse(house))

        featuresData.append(new_data)          

    return normalizer, featuresData, labelsData


class Normalizer:
    def __init__(self, dataframe, features):
        self.means = calcMeans(dataframe, features)
        self.stds = calcStd(dataframe, features)
    
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

    def normalizeHouse(self, house):
        if house == "Gryffindor":
            return [1, 0, 0, 0]
        elif house == "Hufflepuff":
            return [0, 1, 0, 0]
        elif house == "Ravenclaw":
            return [0, 0, 1, 0]
        elif house == "Slytherin":
            return [0, 0, 0, 1]
        
    def denormalizeHouse(self, house):
        if house == [1, 0, 0, 0]:
            return "Gryffindor"
        elif house == [0, 1, 0, 0]:
            return "Hufflepuff"
        elif house == [0, 0, 1, 0]:
            return "Ravenclaw"
        elif house == [0, 0, 0, 1]:
            return "Slytherin"

    def cleanNan(self, dataframe, feature):
        mean = self.means[feature]
        dataframe[feature] = dataframe[feature].fillna(mean)