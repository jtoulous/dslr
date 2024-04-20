import pandas as pd
import sys

from utils.normalizer import Normalizer, normalizeData
from logreg_train import getScores, softmax

def getData():
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    weights = {'Gryffindor': {}, 'Hufflepuff': {}, 'Ravenclaw': {}, 'Slytherin': {}}
    means = {}
    stds = {}
    dataframe = pd.read_csv(sys.argv[1])
    dataframe = dataframe.drop(columns=['Hogwarts House'])
    featuresToUse = []
    featuresToDrop = []

    with open("utils/weights.txt", "r") as weightsFile:
        for line in weightsFile:
            data, features = line.split(':')
            for feature in features.split(','):
                featureName, value = feature.split('=')
                if featureName not in featuresToUse and featureName != 'bias':
                    featuresToUse.append(featureName)

                if data in houses:
                    if featureName != 'Best Hand':
                        weights[data][featureName] = float(value)
                    else:
                        weights[data][featureName] = [float(value.split('/')[0]), float(value.split('/')[1])]            
                elif data == 'Means':
                    means[featureName] = float(value)
                else:
                    stds[featureName] = float(value)
    
    for column in dataframe.columns:
        if column != 'Hogwarts House' and column not in featuresToUse:
            featuresToDrop.append(column)
    dataframe = dataframe.drop(columns=featuresToDrop)
    normalizer, dataset = normalizeData(dataframe, means=means, stds=stds)

    return weights, dataset, normalizer


def initialCheck():
    if (len(sys.argv) != 2):
        raise Exception('error: need a dataset as argument')

if __name__ == "__main__":
    try:
        initialCheck()
        weights, dataset, normalizer = getData()
        scores = getScores(weights, dataset)
        probabilities = softmax(scores)
        data = {
            'Hogwarts House': []
        }

        for prob in probabilities:
            highestProb = 0
            housePredicted = ''
            for house in prob:
                if prob[house] > highestProb:
                    highestProb = prob[house]
                    housePredicted = house
            data['Hogwarts House'].append(housePredicted)
        
        df = pd.DataFrame(data)
        df.to_csv('houses.csv', index_label='Index')


    except Exception as error:
        print(f'{error}')