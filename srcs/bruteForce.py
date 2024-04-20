import sys
import pandas as pd
import numpy as np
import math
from colorama import Fore, Style

from utils.normalizer import normalizeData
from utils.logs import printLog, printInfo, printError
from utils.tools import formatDataframe, checkBestResult, printEpochInfo, endOfTraining, initWeights

def AlreadyTested(featuresToTest, alreadyTested):
    featuresToTestSet = set(featuresToTest)
    for tested in alreadyTested:
        if featuresToTestSet == set(tested):
            return True
    return False

def runTraining(features, records):
    from logreg_train import training
    dataframe = formatDataframe(features)
    normalizer, studentsData = normalizeData(dataframe)
    cost = training(normalizer, studentsData, features, 1)
    if records['bestCost'] is None or cost < records['bestCost']:
        records['bestCost'] = cost
        records['bestFeatures'] = list(features)

def rekMeDaddy(featuresToTest, maxFeatures, records, alreadyTested):
    features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying', 'Best Hand']
    for i in range(14):
        if (len(featuresToTest) == maxFeatures):
            if not AlreadyTested(featuresToTest, alreadyTested):
                runTraining(featuresToTest, records)
                alreadyTested.append(list(featuresToTest))
            return 

        while features[i] in featuresToTest:
            i += 1
            if i == 14:
                return

        featuresToTest.append(features[i])
        rekMeDaddy(featuresToTest, maxFeatures, records, alreadyTested)
        featuresToTest.pop()
        i += 1

def brutForce():
    maxFeatures = 1
    records = {
        'bestCost': None,
        'bestFeatures': []
        }

    while maxFeatures < 14:
        rekMeDaddy([], maxFeatures, records, [])
        maxFeatures += 1
    print(f"minimal cost = {records['bestCost']} ====> {records['bestFeatures']}")