import sys
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style
from utils.logs import printError, printInfo, printLog

def dslrScatter():
    dataframe = pd.read_csv('../datasets/dataset_train.csv')
    featureData1 = dataframe['Defense Against the Dark Arts']
    featureData2 = dataframe['Astronomy']

    gryffData, hufflData, ravenData, slythData = separateData(dataframe, featureData1, featureData2) 
    plt.scatter(gryffData[0], gryffData[1], alpha=0.3, label='Gryffindor', color='darkred')
    plt.scatter(hufflData[0], hufflData[1], alpha=0.3, label='Hufflepuff', color='turquoise')
    plt.scatter(ravenData[0], ravenData[1], alpha=0.3, label='Ravenclaw', color='black')
    plt.scatter(slythData[0], slythData[1], alpha=0.3, label='Slytherin', color='green')

    plt.xlabel('Defense Against the Dark Arts')
    plt.ylabel('Astronomy')
    plt.legend()
    plt.show()

def separateData(dataframe, featureData1, featureData2):
    gryffData, hufflData, ravenData, slythData = [[], []], [[], []], [[], []], [[], []]
    limit = min(len(featureData1), len(featureData2))
    for i in range(limit):
        if dataframe["Hogwarts House"][i] == "Gryffindor":
            gryffData[0].append(featureData1[i])
            gryffData[1].append(featureData2[i])
        elif dataframe["Hogwarts House"][i] == "Hufflepuff":
            hufflData[0].append(featureData1[i])
            hufflData[1].append(featureData2[i])
        elif dataframe["Hogwarts House"][i] == "Ravenclaw":
            ravenData[0].append(featureData1[i])
            ravenData[1].append(featureData2[i])
        elif dataframe["Hogwarts House"][i] == "Slytherin":
            slythData[0].append(featureData1[i])
            slythData[1].append(featureData2[i])
    return gryffData, hufflData, ravenData, slythData

def getTarget2(features, target1):
    featuresCopy = list(features.copy())
    featuresCopy.remove(target1)
    printInfo("\nAvailable features:")
    for i in range(len(featuresCopy)):
        printInfo(f"{i + 1}- {featuresCopy[i]}")
    target2 = input(f"\n{Fore.GREEN}Choose your second feature: {Style.RESET_ALL}")
    
    try:
        target2 = int(target2)
        if target2 > 0 and target2 <= len(featuresCopy):
            return (featuresCopy[target2 - 1])
    except:
        printError(f"No feature named \"{target2}\"")
        target2 = getTarget2(features, target1)
    
    return target2


def getTarget1(features):
    printInfo("\nAvailable features:")
    for i in range(len(features)):
        printInfo(f"{i + 1}- {features[i]}")
    target1 = input(f"\n{Fore.GREEN}Choose your first feature: {Style.RESET_ALL}")
    
    try: 
        target1 = int(target1)
        if target1 > 0 and target1 <= len(features):
            return features[target1 - 1]
    except ValueError:
        printError(f"No feature named \"{target1}\"")
        target1 = getTarget1(features) 
    
    return target1


def getTargets(features):
    target1 = getTarget1(features)
    target2 = getTarget2(features, target1)
    return target1, target2


def buildScatter():
    dataframe = pd.read_csv(sys.argv[1])
    features = dataframe.select_dtypes(include=["float64"]).columns
    target1, target2 = getTargets(features)
    featureData1 = dataframe[target1]
    featureData2 = dataframe[target2]

################################################    
########    Choose scatter options    #########
       ################################
            #####################
                #############
                    #####
                      #

    gryffData, hufflData, ravenData, slythData = separateData(dataframe, featureData1, featureData2) 
    plt.scatter(gryffData[0], gryffData[1], alpha=0.3, label='Gryffindor', color='darkred')
    plt.scatter(hufflData[0], hufflData[1], alpha=0.3, label='Hufflepuff', color='turquoise')
    plt.scatter(ravenData[0], ravenData[1], alpha=0.3, label='Ravenclaw', color='black')
    plt.scatter(slythData[0], slythData[1], alpha=0.3, label='Slytherin', color='green')

    plt.xlabel(target1)
    plt.ylabel(target2)
    plt.legend()
################################################    
    plt.show()


if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            dslrScatter()
        else:
            buildScatter()

    except Exception as error:
        printError(error)