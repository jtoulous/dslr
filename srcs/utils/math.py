import math

def getMax(dataList):
    max = dataList[0]
    for data in dataList:
        max = data if data > max else max
    return max

def getPercentile(dataList, percent):
    length = 0
    sorted = [] 
    for data in dataList:
        if not math.isnan(data):
            length += 1
            sorted.append(data)

    sorted.sort()
    idx = round(length * (percent / 100))
    if idx == 0:
        idx = 1
    
    if length == 0 and len(dataList) != 0:
        return math.nan
    elif length == 0:
        return 0
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
    if length != 0:
        return float((variance / length) ** 0.5)
    elif length == 0 and len(dataList) != 0:
        return math.nan 
    return 0

def getMean(dataList):
    length = 0
    total = 0
    for data in dataList:
        if not math.isnan(data):
            total += data
            length += 1
    if length != 0:
        return float(total / length)
    elif length == 0 and len(dataList) != 0:
        return math.nan 
    return 0

def getLength(dataList):
    length = 0
    for data in dataList:
        if not math.isnan(data):
            length += 1
    return length