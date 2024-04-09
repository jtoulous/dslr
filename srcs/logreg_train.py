import sys
import pandas as pd
import numpy as np

from utils.normalizer import normalizeData

def initialCheck():
    if len(sys.argv) != 2:
        raise Exception("Error: data file needed as argument")


def initWeights(studentsData):
    weightsData = {
        'gryff': {'weights': [], 'bestHandWeight':[] , 'bias': 0},
        'huffl': {'weights': [], 'bestHandWeight':[] , 'bias': 0},
        'slyth': {'weights': [], 'bestHandWeight':[] , 'bias': 0},
        'raven': {'weights': [], 'bestHandWeight':[] , 'bias': 0}
    }

    for house in weightsData:
        for i in range(len(studentsData[0]) - 1):
            weightsData[house]['weights'].append(np.random.uniform(-1, 1))
        weightsData[house]['bestHandWeight'].append(np.random.uniform(-1, 1))
        weightsData[house]['bestHandWeight'].append(np.random.uniform(-1, 1))

    return weightsData


def getScores(weightsData, studentsData):
    scores = [] #list of dict size 4
    houses = ['gryff', 'huffl', 'slyth', 'raven']

    for student in studentsData:
        studentScores = {}
        for house in houses:
            weightsHouse = weightsData[house]
            score = 0
            i = 0
            for feature in student:
                if feature == 'Best Hand':
                    if student[feature] == [1, 0]:
                        score += weightsHouse['bestHandWeight'][0]       
                    else:
                        score += weightsHouse['bestHandWeight'][1]       
                else:
                    score += student[feature] * weightsHouse['weights'][i]
                i += 1
            studentScores[house] = score + weightsHouse['bias']
        scores.append(studentScores)
    return scores

def getProbabilities(scores):
    probabilities = []
    houses = ['gryff', 'huffl', 'slyth', 'raven']
    
    for studentScores in scores:
        studentProbs = {}
        for house in houses:
            newProb = 1 / (1 + np.exp(-studentScores[house]))
            studentProbs[house] = newProb
        probabilities.append(studentProbs)
    return probabilities

def training(normalizer, studentsData, labels):
    epochs = 300
    learningRate = 0.01
    weightsData = initWeights(studentsData)

    for i in range(epochs):
        scores = getScores(weightsData, studentsData)
        probabilities = getProbabilities(scores)
        #entropicCosts = getCost(probabilities, labels, studentsData)
        #gradientDescent(learningRate, entropicCosts, probabilities, labels, studentsData)
        

if __name__ == "__main__":
    try:
        initialCheck()
        dataframe = pd.read_csv(sys.argv[1])
        normalizer, studentsData, labelsData = normalizeData(dataframe)
        training(normalizer, studentsData, labelsData)

    except Exception as error:
        print(error)