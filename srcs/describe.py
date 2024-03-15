import sys
from statistics import mean
import pandas as pd

def getMax(dataList):
    max = dataList[0]
    for data in dataList:
        max = data if data > max else max
    return max

def getPercentile(dataList, percent):
    sorted = list(dataList.copy()) 
    sorted.sort()
    idx = round(len(sorted) * (percent / 100))
    return sorted[idx - 1]

def getMin(dataList):
    min = dataList[0]
    for data in dataList:
        min = data if data < min else min
    return min

def getStd(dataList):
    mean = getMean(dataList)
    variance = sum((data - mean) ** 2 for data in dataList)
    return float((variance / (len(dataList) - 1)) ** 0.5)

def getMean(dataList):
    print (mean(dataList))
    return float(sum(dataList) / len(dataList)) 

def checkArgv():
    args = sys.argv
    if (len(args) != 2):
        raise Exception('Error: one argument required')
    return args[1]

if __name__ == "__main__":
    try:    
        filePath = checkArgv()
        dataset = pd.read_csv(filePath)
        features = dataset.select_dtypes(include=["float64"]).columns
        stats = []

        for feature in features:
            featureData = dataset[feature]
            stats.append([
                feature[:10],
                float(len(featureData)),
                getMean(featureData),
                getStd(featureData),
                getMin(featureData),
                getPercentile(featureData, 25),
                getPercentile(featureData, 50),
                getPercentile(featureData, 75),
                getMax(featureData)
            ])
        
        dataframe = pd.DataFrame(stats, columns=["Feature", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
        transposed = dataframe.transpose()
        print(f"\n{transposed.to_string(index=True, header=False)}\n")

    except Exception as error:
        print(error)