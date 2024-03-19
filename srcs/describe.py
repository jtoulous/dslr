import sys
import math
import pandas as pd
import numpy as np
from utils.math import getMean, getLength, getStd, getMin, getPercentile, getMax

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
                feature[:15],
                round(getLength(featureData), 6),
                round(getMean(featureData), 6),
                round(getStd(featureData), 6),
                round(getMin(featureData), 6),
                round(getPercentile(featureData, 25), 6),
                round(getPercentile(featureData, 50), 6),
                round(getPercentile(featureData, 75), 6),
                round(getMax(featureData), 6)
            ])
        
        dataframe = pd.DataFrame(stats, columns=["", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
        print(f"\n{dataframe.to_string(index=False, col_space=10)}\n")
        
        #transposed = dataframe.transpose()
        #print(f"\n{transposed.to_string(index=True, header=False)}\n")

    except Exception as error:
        print(error)