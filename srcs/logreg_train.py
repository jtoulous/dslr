import sys
import pandas as pd
import numpy as np
import math
import colorama

from utils.normalizer import normalizeData
from utils.logs import printLog, printInfo, printError
from brutForce import brutForce

def initialCheck():
    if len(sys.argv) != 2:
        raise Exception("Error: data file needed as argument")


def formatDataframe(brutForceFeatures=None):
    dataframe = pd.read_csv(sys.argv[1])
    dataframe = dataframe.drop(columns=["Index", "First Name", "Last Name", "Birthday"])

######################################################################    
    if brutForceFeatures is not None:
        tmpDataframe = dataframe.copy()
        featuresToDrop = [feat for feat in tmpDataframe.columns if feat not in brutForceFeatures and feat != 'Hogwarts House']
        dataframe = dataframe.drop(columns=featuresToDrop)
        return dataframe
#######################################################################

    tmpDataframe = dataframe.copy()
    chosenFeatures = []
    done = 0
    
    tmpDataframe = tmpDataframe.drop(columns=["Hogwarts House"])
    while True:    
        print('')
        for i, feature in enumerate(tmpDataframe.columns):     
            printInfo(f'{i}: {feature}')
        printLog('\nfinished = \'done\'\n')
        
        entry = input('make your choice: ')
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


def initWeights(studentsData, features):
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    weights = {
        'Gryffindor': {},
        'Hufflepuff': {},
        'Ravenclaw': {},
        'Slytherin': {}
    }
    for house in houses:    
        for feature in features:
            if feature == 'Best Hand':
                weights[house][feature] = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
            else:
                weights[house][feature] = np.random.uniform(-1, 1)
        weights[house]['bias'] = 0
    return weights


def getScores(weights, studentsData):
    scores = []
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    for student in studentsData:
        studentScores = {}
        for house in houses:
            houseWeights = weights[house]
            score = 0
            i = 0
            for feature in student:
                if feature == 'Best Hand':
                    if student[feature] == [1, 0]:
                        score += houseWeights[feature][0]       
                    else:
                        score += houseWeights[feature][1]       
                else:
                    score += student[feature] * houseWeights[feature]
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


def getCost(probabilities, labels, normalizer):
    totalError = 0
    
    for i in range(len(probabilities)):
        trueHouse = normalizer.denormalizeHouse(labels[i])
        prediction = probabilities[i][trueHouse]
        totalError += math.log(prediction)
    return -(totalError / len(probabilities))


def gradientDescent(weights, learningRate, meanCosts, probabilities, labels, studentsData, features):
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    
    for house in houses:
        houseWeights = weights[house]
        for feature in houseWeights:
            if feature == 'Best Hand' or feature == 'bias':
                continue
            totalGradient = 0
            for i, student in enumerate(studentsData):
                y = 1 if house == labels[i] else 0
                featureVal = student[feature]
                prob = probabilities[i][house]
                totalGradient += (prob - y) * featureVal
            gradient = totalGradient / len(studentsData)
            houseWeights[feature] -= learningRate * gradient
    
    if 'Best Hand' in features:
        for house in houses:
            houseWeights = weights[house]
            totalGradientLeft = 0
            totalGradientRight = 0
            countLeft = 0
            countRight = 0
            for i, student in enumerate(studentsData):
                y = 1 if house == labels[i] else 0
                featureVal = student['Best Hand']
                prob = probabilities[i][house]

                if student['Best Hand'] == [1, 0]:
                    totalGradientLeft += prob - y
                    countLeft += 1
                else:
                    totalGradientRight += prob - y
                    countRight += 1
            gradientLeft = totalGradientLeft / countLeft
            gradientRight = totalGradientRight / countRight
            houseWeights['Best Hand'][0] -= learningRate * gradientLeft
            houseWeights['Best Hand'][1] -= learningRate * gradientRight
    
    for house in houses:
        houseWeights = weights[house]
        totalBiasGradient = 0
        for i, student in enumerate(studentsData):
            y = 1 if house == labels[i] else 0
            prob = probabilities[i][house]
            totalBiasGradient += prob - y
        biasGradient = totalBiasGradient / len(studentsData)
        houseWeights['bias'] -= learningRate * biasGradient 


def printPredictions(probabilities, labels, normalizer):
    for i in range(len(labels)):
        prediction = ''
        highestProb = 0
        for house in probabilities[i]:
            prob = probabilities[i][house]
            if prob > highestProb:
                prediction = house
                highestProb = prob
        print(f'{normalizer.denormalizeHouse(labels[i])} ====> {prediction}')


def training(normalizer, studentsData, labels, features, brutForce=0):
    epochs = 200
    learningRate = 0.7
    weights = initWeights(studentsData, features)

    for i in range(epochs):
        scores = getScores(weights, studentsData)
        probabilities = softmax(scores)
        meanCost = getCost(probabilities, labels, normalizer)
        gradientDescent(weights, learningRate, meanCost, probabilities, labels, studentsData, features)
    
        print(f'\n==========   Epoch {i}  ========')
        print(f'Cost ====> {meanCost}')
        if i == 499:    
            printPredictions(probabilities, labels, normalizer)
    
    if brutForce == 0:
        return weights
    else:
        return meanCost

#########################################################

def runTraining(features, records):
    dataframe = formatDataframe(features)
    normalizer, studentsData, labels = normalizeData(dataframe)
    cost = training(normalizer, studentsData, labels, features, 1)
    if records['bestCost'] is None or cost < records['bestCost']:
        records['bestCost'] = cost
        records['bestFeatures'] = list(features)

def rekMeDaddy(featuresToTest, maxFeatures, records):
    features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying', 'Best Hand']
    for i in range(14):
        if (len(featuresToTest) == maxFeatures):
            runTraining(featuresToTest, records)
            return 

        while features[i] in featuresToTest:
            i += 1
            if i == 14:
                return

        featuresToTest.append(features[i])
        rekMeDaddy(featuresToTest, maxFeatures, records)
        featuresToTest.pop()
        i += 1

def brutForce():
    bestFeatures = []
    bestCost = None
    maxFeatures = 1
    records = {
        'bestCost': None,
        'bestFeatures': []
        }

    while maxFeatures < 14:
        rekMeDaddy([], maxFeatures, records)
        maxFeatures += 1
    print(f"minimal cost = {records['bestCost']} ====> {records['bestFeatures']}")

##################################################################



if __name__ == "__main__":
    try:
        initialCheck()
        brutForce()
        #dataframe, features = formatDataframe()
        #normalizer, studentsData, labels = normalizeData(dataframe)
        #training(normalizer, studentsData, labels, features)
        #for feature in features:
        #    print(f'\n{feature}')

#brutForce()

    except Exception as error:
        print(error)