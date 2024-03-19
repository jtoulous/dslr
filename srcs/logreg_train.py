import sys
import pandas as pd

from utils.normalizer import normalizeData

def initialCheck():
    if len(sys.argv) != 2:
        raise Exception("Error: data file needed as argument")


if __name__ == "__main__":
    try:
        initialCheck()
        dataset = normalizeData()


    except Exception as error:
        print(error)