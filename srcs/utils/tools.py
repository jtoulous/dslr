import sys
import pandas as pd
import numpy as np
import math
import copy
from colorama import Fore, Style

from .logs import printLog, printInfo, printError
from .normalizer import normalizeData, Normalizer

def getLastFeatures():
    featuresList = []
    with open('utils/weights.txt', 'r') as weightsFile:
        for line in weightsFile:
            features = line.split(':')[1]
            for feature in features.split(','):
                featureName = feature.split('=')[0]
                if featureName != 'bias':
                    featuresList.append(featureName)
            return featuresList


def initialCheck():
    if len(sys.argv) < 2:
        raise Exception("Error: data file needed as argument")


def formatDataframe(brutForceFeatures=None):
    dataframe = pd.read_csv(sys.argv[1])
    dataframe = dataframe.drop(columns=["Index", "First Name", "Last Name", "Birthday"])
    tmpDataframe = dataframe.copy()

    if brutForceFeatures is not None:
        featuresToDrop = [feat for feat in tmpDataframe.columns if feat not in brutForceFeatures and feat != 'Hogwarts House']
        dataframe = dataframe.drop(columns=featuresToDrop)
        return dataframe

    chosenFeatures = []
    done = 0

    retest = input(f'{Fore.GREEN}Retry last used features?\n\'yes\' or \'no\': {Style.RESET_ALL}')
    if retest == 'yes':
        retestFeatures = getLastFeatures()
        featuresToDrop = [feat for feat in tmpDataframe.columns if feat not in retestFeatures and feat != 'Hogwarts House']
        dataframe = dataframe.drop(columns=featuresToDrop)
        return dataframe, retestFeatures
    
    tmpDataframe = tmpDataframe.drop(columns=["Hogwarts House"])
    while True:    
        print('')
        for i, feature in enumerate(tmpDataframe.columns):     
            printInfo(f'{i}: {feature}')
        printInfo('Finished : \'done\'\n')
        
        entry = input(f'{Fore.GREEN}make your choice: {Style.RESET_ALL}')

        if entry == 'done':
            tmpDataframe = dataframe.copy()
            featuresToDrop = [feat for feat in tmpDataframe.columns if feat not in chosenFeatures and feat != 'Hogwarts House']
            dataframe = dataframe.drop(columns=featuresToDrop)
            return dataframe, chosenFeatures
        
        entry = int(entry)
        if entry >= len(tmpDataframe.columns):
            printError(f'feature {entry} not available')
        
        chosenfeat = tmpDataframe.columns[entry]
        chosenFeatures.append(chosenfeat)
        tmpDataframe = tmpDataframe.drop(columns=[chosenfeat])


def printPredictions(probabilities, studentsData):
    answer = input(f'\n{Fore.GREEN}Print prediction, \'yes\' or \'no\': {Style.RESET_ALL}')
    if answer == 'yes' or answer == 'YES':
        correctPredictCount = 0
        for i in range(len(studentsData)):
            trueHouse = studentsData[i]['label']
            prediction = ''
            highestProb = 0
            for house in probabilities[i]:
                prob = probabilities[i][house]
                if prob > highestProb:
                    prediction = house
                    highestProb = prob
            if trueHouse == prediction:
                printLog(f'{trueHouse} ======> {prediction}')
                correctPredictCount += 1
            else:
                printError(f'{trueHouse} ======> {prediction}')
        correctPredictPercentage = int((correctPredictCount / len(studentsData)) * 100)
        printLog(f'\n=====> {correctPredictPercentage}% successful')

def saveWeight(weights):
    with open('utils/weights.txt', 'w') as weightsFile:
        for house in weights:
            houseWeights = weights[house]
            weightsFile.write(f'{house}:')
            for feature in houseWeights:
                if feature != 'bias' and feature != 'Best Hand':
                    weightsFile.write(f'{feature}={houseWeights[feature]},')
                elif feature == 'Best Hand':
                    weightsFile.write(f'{feature}={houseWeights[feature][0]}/{houseWeights[feature][1]},')
                else:
                    weightsFile.write(f'{feature}={houseWeights[feature]}')
            weightsFile.write(f'\n')

def AlreadyTested(featuresToTest, alreadyTested):
    featuresToTestSet = set(featuresToTest)
    for tested in alreadyTested:
        if featuresToTestSet == set(tested):
            return True
    return False

def printGradients(gradients):
    for house in gradients:
        houseGradient = gradients[house]
        i = 0
        print(f'{Fore.RED}{house}{Style.RESET_ALL}: ', end='')
        for feature, gradient in houseGradient.items():
            if i == len(houseGradient) - 1:
                print(f'{Fore.BLUE}{feature}{Style.RESET_ALL} = {gradient}', end='')
            else:
                print(f'{Fore.BLUE}{feature}{Style.RESET_ALL} = {gradient}, ', end='')
            i += 1
        print()

def printEpochInfo(i, meanCost, gradients):
            print(f'\n==========   Epoch {i}  ========')
            print(f'Cost ====> {meanCost}')
            printGradients(gradients)

def endOfTraining(features, weights, studentsData, bestResults, brutForce):
    if brutForce == None:
        printPredictions(bestResults['bestProbs'], studentsData)
    printInfo(f'\n===> Used features : {", ".join(features)}')
    printInfo(f'===> Best cost = {bestResults["bestCost"]}')
    
    if brutForce == None:
        saveWeight(bestResults['bestWeights'])

def checkBestResult(bestResults, meanCost, weights, probabilities):
    if bestResults['bestCost'] is None or meanCost < bestResults['bestCost']:
        bestResults['bestCost'] = meanCost
        bestResults['bestWeights'] = copy.deepcopy(weights)
        bestResults['bestProbs'] = copy.deepcopy(probabilities)

def initWeights(features):
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    weights = {
        'Gryffindor': {},
        'Hufflepuff': {},
        'Ravenclaw': {},
        'Slytherin': {}
    }
    stddev = 0.01
    for house in houses:    
        for feature in features:
            if feature == 'Best Hand':
                weights[house][feature] = [np.random.normal(0, stddev), np.random.normal(0, stddev)]
            else:
                weights[house][feature] = np.random.normal(0, stddev)
        weights[house]['bias'] = np.random.normal(0, stddev)
    return weights