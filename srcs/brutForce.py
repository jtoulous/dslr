def runTraining(features, records):
    dataframe = formatDataframe(features)
    normalizer, studentsData, labels = normalizeData(dataframe)
    cost = training(normalizer, studentsData, labels, features, 1)
    if records['bestCost'] is None or cost < records['bestCost']:
        records['bestCost'] = cost
        records['bestFeatures'] = list(features)

def rekMeDaddy(featuresToTest, maxFeatures, records):
    features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying', 'Best Hand']
    for i in range(14):
        if (len(featuresToTest) == maxFeatures):
            runTraining(featuresToTest, records)
            return 

        while features[i] in featuresToTest:
            i += 1
            if i == 14:
                return

        featuresToTest.append(features[i])
        rekMeDaddy(featuresToTest, maxFeatures, records)
        featuresToTest.pop()
        i += 1

def brutForce():
    bestFeatures = []
    bestCost = None
    maxFeatures = 1
    records = {
        'bestCost': None,
        'bestFeatures': []
        }

    while maxFeatures < 14:
        rekMeDaddy([], maxFeatures, records)
        maxFeatures += 1
    print(f"minimal cost = {records['bestCost']} ====> {records['bestFeatures']}")