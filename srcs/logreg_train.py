import sys
import pandas as pd
import numpy as np
import math

from utils.normalizer import normalizeData

def initialCheck():
    if len(sys.argv) != 2:
        raise Exception("Error: data file needed as argument")


def initWeights(studentsData):
    weights = {
        'Gryffindor': {
            'Arithmancy': 0,
            'Astronomy': 0,
            'Herbology': 0,
            'Defense Against the Dark Arts': 0,
            'Divination': 0,
            'Muggle Studies': 0,
            'Ancient Runes': 0,
            'History of Magic': 0,
            'Transfiguration': 0,
            'Potions': 0,
            'Care of Magical Creatures': 0,
            'Charms': 0,
            'Flying': 0,
            'Best Hand': [],
            'bias': 0
            },
        
        'Hufflepuff': {
            'Arithmancy': 0,
            'Astronomy': 0,
            'Herbology': 0,
            'Defense Against the Dark Arts': 0,
            'Divination': 0,
            'Muggle Studies': 0,
            'Ancient Runes': 0,
            'History of Magic': 0,
            'Transfiguration': 0,
            'Potions': 0,
            'Care of Magical Creatures': 0,
            'Charms': 0,
            'Flying': 0,
            'Best Hand': [],
            'bias': 0
            },
        
        'Ravenclaw': {
            'Arithmancy': 0,
            'Astronomy': 0,
            'Herbology': 0,
            'Defense Against the Dark Arts': 0,
            'Divination': 0,
            'Muggle Studies': 0,
            'Ancient Runes': 0,
            'History of Magic': 0,
            'Transfiguration': 0,
            'Potions': 0,
            'Care of Magical Creatures': 0,
            'Charms': 0,
            'Flying': 0,
            'Best Hand': [],
            'bias': 0
            },
        
        'Slytherin': {
            'Arithmancy': 0,
            'Astronomy': 0,
            'Herbology': 0,
            'Defense Against the Dark Arts': 0,
            'Divination': 0,
            'Muggle Studies': 0,
            'Ancient Runes': 0,
            'History of Magic': 0,
            'Transfiguration': 0,
            'Potions': 0,
            'Care of Magical Creatures': 0,
            'Charms': 0,
            'Flying': 0,
            'Best Hand': [],
            'bias': 0
            }
    }

    for house in weights:
        for feature in weights[house]:
            if feature == 'Best Hand':
                weights[house][feature].append(np.random.uniform(-1, 1))
                weights[house][feature].append(np.random.uniform(-1, 1))
            else:
                weights[house][feature] = np.random.uniform(-1, 1)

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


def gradientDescent(weights, learningRate, meanCosts, probabilities, labels, studentsData):
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


def training(normalizer, studentsData, labels):
    epochs = 500
    learningRate = 0.7
    weights = initWeights(studentsData)

    for i in range(epochs):
        scores = getScores(weights, studentsData)
        probabilities = softmax(scores)
        meanCost = getCost(probabilities, labels, normalizer)
        gradientDescent(weights, learningRate, meanCost, probabilities, labels, studentsData)
    
        print(f'\n==========   Epoch {i}  ========')
        print(f'Cost ====> {meanCost}')
        if i == 499:    
            printPredictions(probabilities, labels, normalizer)

        

if __name__ == "__main__":
    try:
        initialCheck()
        dataframe = pd.read_csv(sys.argv[1])
        normalizer, studentsData, labelsData = normalizeData(dataframe)
        training(normalizer, studentsData, labelsData)

    except Exception as error:
        print(error)