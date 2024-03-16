import sys
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style
from utils.logs import printError, printInfo, printLog

def dslrHisto():
    dataframe = pd.read_csv("../datasets/dataset_train.csv")
    featureData = dataframe["Care of Magical Creatures"]

    gryffData, hufflData, ravenData, slythData = sepData(dataframe, featureData)
    plt.hist(gryffData, alpha=0.4, label='Gryffindor', color='darkred')
    plt.hist(hufflData, alpha=0.4, label='Hufflepuff', color='turquoise')
    plt.hist(ravenData, alpha=0.4, label='Ravenclaw', color='black')
    plt.hist(slythData, alpha=0.4, label='Slytherin', color='green')

    plt.xlabel("Grades")
    plt.ylabel("Frequency")
    plt.title(f"Care of Magical Creatures")
    plt.legend()
    plt.show()


def separateData(dataframe, featureData):
    gryffData, hufflData, ravenData, slythData = [], [], [], []
    
    for i in range(len(featureData)):
        if dataframe["Hogwarts House"][i] == "Gryffindor":
            gryffData.append(featureData[i])
        elif dataframe["Hogwarts House"][i] == "Hufflepuff":
            hufflData.append(featureData[i])
        elif dataframe["Hogwarts House"][i] == "Ravenclaw":
            ravenData.append(featureData[i])
        elif dataframe["Hogwarts House"][i] == "Slytherin":
            slythData.append(featureData[i])
    return gryffData, hufflData, ravenData, slythData


def getTarget(features):
    printInfo("\nAvailable features:")
    for feature in features:
        printInfo(f"-{feature}")
    target = input(f"\n{Fore.GREEN}Choose your feature: {Style.RESET_ALL}")
    
    for feature in features:
        if target == feature:
            return target
    printError(f"No feature named \"{target}\"")
    target = getTarget(features)
    return target


def buildHisto():
    dataframe = pd.read_csv(sys.argv[1])
    features = dataframe.select_dtypes(include=["float64"]).columns    
    target = getTarget(features)
    featureData = dataframe[target]

################################################    
########    Choose histogram options   #########
       ################################
            #####################
                #############
                    #####
                      #
    gryffData, hufflData, ravenData, slythData = separateData(dataframe, featureData)
    plt.hist(gryffData, alpha=0.4, label='Gryffindor', color='darkred')
    plt.hist(hufflData, alpha=0.4, label='Hufflepuff', color='turquoise')
    plt.hist(ravenData, alpha=0.4, label='Ravenclaw', color='black')
    plt.hist(slythData, alpha=0.4, label='Slytherin', color='green')

    plt.xlabel("Grades")
    plt.ylabel("Frequency")
    plt.title(f"{target}")
    plt.legend()
################################################    
    plt.show()


if __name__ == "__main__":
    try:
        args = sys.argv
        if len(args) == 1:
            dslrHisto()
        else:
            buildHisto()

    except Exception as error:
        print(error)