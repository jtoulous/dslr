import sys
import pandas as pd
import numpy as np
import math
from colorama import Fore, Style

from utils.normalizer import normalizeData
from utils.logs import printLog, printInfo, printError
from utils.tools import getLastFeatures, initialCheck, formatDataframe, checkBestResult, printEpochInfo, endOfTraining, initWeights
from bruteForce import brutForce

def getScores(weights, studentsData):
    scores = []
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    for student in studentsData:
        studentScores = {}
        for house in houses:
            studentFeatures = student['features']
            houseWeights = weights[house]
            score = 0
            i = 0
            for feature in studentFeatures:
                if feature == 'Best Hand':
                    if studentFeatures[feature] == [1, 0]:
                        score += houseWeights[feature][0]       
                    else:
                        score += houseWeights[feature][1]       
                else:
                    score += studentFeatures[feature] * houseWeights[feature]
                i += 1
            studentScores[house] = score + houseWeights['bias']
        scores.append(studentScores)
    return scores


def softmax(scores):
    probabilities = []
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    for studentScores in scores:
        studentProbs = {}
        expTotalScore = sum(np.exp(score) for score in studentScores.values())
        
        for house in houses:
            newProb = np.exp(studentScores[house]) / expTotalScore
            studentProbs[house] = newProb
        probabilities.append(studentProbs)
    return probabilities


def getCost(probabilities, studentsData, normalizer):
    totalError = 0
    
    for i in range(len(probabilities)):
        trueHouse = studentsData[i]['label']
        prediction = probabilities[i][trueHouse]
        totalError += math.log(prediction)
    return -(totalError / len(probabilities))


def gradientDescent(weights, learningRate, probabilities, studentsData, features):
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    gradientData = {
        'Gryffindor': {},
        'Hufflepuff': {},
        'Ravenclaw': {},
        'Slytherin': {}
    }

    for house in houses:
        houseWeights = weights[house]
        for feature in houseWeights:
            if feature == 'Best Hand' or feature == 'bias':
                continue
            totalGradient = 0
            for i, student in enumerate(studentsData):
                y = 1 if house == student['label'] else 0
                featureVal = student['features'][feature]
                prob = probabilities[i][house]
                totalGradient += (prob - y) * featureVal
            gradient = totalGradient / len(studentsData)
            houseWeights[feature] -= learningRate * gradient
            gradientData[house][feature] = gradient
    
    if 'Best Hand' in features:
        for house in houses:
            houseWeights = weights[house]
            totalGradientLeft = 0
            totalGradientRight = 0
            countLeft = 0
            countRight = 0
            for i, student in enumerate(studentsData):
                y = 1 if house == student['label'] else 0
                featureVal = student['features']['Best Hand']
                prob = probabilities[i][house]

                if featureVal == [1, 0]:
                    totalGradientLeft += prob - y
                    countLeft += 1
                else:
                    totalGradientRight += prob - y
                    countRight += 1
            gradientLeft = totalGradientLeft / countLeft
            gradientRight = totalGradientRight / countRight
            houseWeights['Best Hand'][0] -= learningRate * gradientLeft
            houseWeights['Best Hand'][1] -= learningRate * gradientRight
            gradientData[house]['left'] = gradientLeft
            gradientData[house]['right'] = gradientRight
    
    for house in houses:
        houseWeights = weights[house]
        totalBiasGradient = 0
        for i, student in enumerate(studentsData):
            y = 1 if house == student['label'] else 0
            prob = probabilities[i][house]
            totalBiasGradient += prob - y
        biasGradient = totalBiasGradient / len(studentsData)
        houseWeights['bias'] -= learningRate * biasGradient 
    return gradientData


def training(normalizer, studentsData, features, brutForce=None):
    epochs = 300
    learningRate = 0.3
    weights = initWeights(features)
    bestResults = {'bestWeights': {}, 'bestCost': None, 'bestProbs': []}

    for i in range(epochs):
        scores = getScores(weights, studentsData)
        probabilities = softmax(scores)
        meanCost = getCost(probabilities, studentsData, normalizer)
        checkBestResult(bestResults, meanCost, weights, probabilities)

        gradients = gradientDescent(weights, learningRate, probabilities, studentsData, features)
        if brutForce == None:
            printEpochInfo(i, meanCost, gradients)

    endOfTraining(features, weights, studentsData, bestResults, normalizer, brutForce)
    return (bestResults['bestCost'])


if __name__ == "__main__":
    try:
        initialCheck()
        if len(sys.argv) > 2 and sys.argv[2] == "--bf":
            brutForce()
        else:
            dataframe, features = formatDataframe()
            normalizer, studentsData = normalizeData(dataframe)
            training(normalizer, studentsData, features)

    except Exception as error:
        print(error)