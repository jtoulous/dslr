import sys
import pandas as pd
import matplotlib.pyplot as plt
from utils.logs import printError, printInfo, printLog

def separateData(dataframe=None, featureData1=None, featureData2=None, plotType=None):
    if plotType == "histo":
        gryffData, hufflData, ravenData, slythData = [], [], [], []
    
        for i in range(len(featureData1)):
            if dataframe["Hogwarts House"][i] == "Gryffindor":
                gryffData.append(featureData1[i])
            elif dataframe["Hogwarts House"][i] == "Hufflepuff":
                hufflData.append(featureData1[i])
            elif dataframe["Hogwarts House"][i] == "Ravenclaw":
                ravenData.append(featureData1[i])
            elif dataframe["Hogwarts House"][i] == "Slytherin":
                slythData.append(featureData1[i])

    else:
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


def buildPair():
    dataframe = pd.read_csv("datasets/dataset_train.csv")
    features = dataframe.select_dtypes(include=["float64"]).columns
    labels = [feature.split(' ')[0] for feature in features]

    fig, axs = plt.subplots(nrows=len(features), ncols=len(features), figsize=(45, 40))

    for x in range(len(features)):
        for y in range(len(features)):
            if x == len(features) - 1:
                axs[x, y].set_xlabel(f"{labels[y]}", fontsize=10)
            if y == 0:
                axs[x, y].set_ylabel(f"{labels[x]}", fontsize=10, rotation=0, ha='right')

########################################################################################    
                    ########    Choose pair options    #########
                           ################################
                                #####################
                                    #############
                                        #####
                                          #
            if x == y:
                gryffData, hufflData, ravenData, slythData = separateData(dataframe=dataframe, featureData1=dataframe[features[x]], plotType="histo")
                axs[x, y].hist(gryffData, alpha=0.3, label='Gryffindor', color='darkred')
                axs[x, y].hist(hufflData, alpha=0.3, label='Hufflepuff', color='turquoise')
                axs[x, y].hist(ravenData, alpha=0.3, label='Ravenclaw', color='black')
                axs[x, y].hist(slythData, alpha=0.3, label='Slytherin', color='green')
                axs[x, y].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            
            else:
                gryffData, hufflData, ravenData, slythData = separateData(dataframe=dataframe, featureData1=dataframe[features[x]], featureData2=dataframe[features[y]], plotType="scatter")
                axs[x, y].scatter(gryffData[0], gryffData[1], s=2, alpha=0.4, color='darkred')
                axs[x, y].scatter(hufflData[0], hufflData[1], s=2, alpha=0.4, color='turquoise')
                axs[x, y].scatter(ravenData[0], ravenData[1], s=2, alpha=0.4, color='black')
                axs[x, y].scatter(slythData[0], slythData[1], s=2, alpha=0.4, color='green')
                axs[x, y].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
##############################################################################                
    plt.show()

if __name__ == "__main__":
    try:
        buildPair()

    except Exception as error:
        printError(error)