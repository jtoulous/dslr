import sys
import pandas as pd

from utils.normalizer import normalizeData

def initialCheck():
    if len(sys.argv) != 2:
        raise Exception("Error: data file needed as argument")

def genRandomWeights(featuresData):
    

def training(normalizer, featuresData, labels):
    weights = genRandomWeights(featuresData)

if __name__ == "__main__":
    try:
        initialCheck()
        dataframe = pd.read_csv(sys.argv[1])
        normalizer, featuresData, labelsData = normalizeData(dataframe)
        print(featuresData[0])
        training(normalizer, featuresData, labelsData)

    except Exception as error:
        print(error)