import sys
import math

import pandas as pd
import numpy as np

def getMax(dataList):
    max = dataList[0]
    for data in dataList:
        max = data if data > max else max
    return max

def getPercentile(dataList, percent):
    length = 0
    sorted = list(dataList.copy()) 
    sorted.sort()
    
    for data in sorted:
        if not math.isnan(data):
            length += 1
    idx = round(length * (percent / 100))
    return sorted[idx - 1]

def getMin(dataList):
    min = dataList[0]
    for data in dataList:
        min = data if data < min else min
    return min

def getStd(dataList):
    mean = getMean(dataList)
    variance = 0
    length = 0
    for data in dataList:
        if not math.isnan(data):
            variance += (data - mean) ** 2 
            length += 1
    return float((variance / length) ** 0.5)

def getMean(dataList):
    lenght = 0
    total = 0
    for data in dataList:
        if not math.isnan(data):
            total += data
            lenght += 1
    return round(float(total / lenght), 3) 

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
        
        dataframe = pd.DataFrame(stats, columns=["", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
        transposed = dataframe.transpose()
        print(f"\n{transposed.to_string(index=True, header=False)}\n")

    except Exception as error:
        print(error)