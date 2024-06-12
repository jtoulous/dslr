## Project Objectives
Train a model using logistic regression to predict the Hogwarts house of each student in the data files.

1. Run `make` to create and install dependencies.
2. Activate the virtual environment (`. virtualEnv/bin/activate`).
3. Run `python logreg_train.py ../datasets/dataset_train.csv` to train the model. You can choose the features that you want to use or just type 'all' to get the best results.
4. Run `python logreg_predict.py ../datasets/dataset_test.csv` to run predictions on the data file, the result will be outputed in a file called 'houses.txt'.

### Bonus:
1. Run `python logreg_train.py ../datasets/dataset_train.csv --bf` to see the final results of the model using every possible combination of features.
2. Run `python histogram.py` to see the required histogram, or `python histogram.py ../datasets/dataset_train.csv` to select the features used in the histogram.
3. Run `python scatter_plot.py` to see the required scatter plot, or `python scatter_plot.py ../datasets/dataset_train.csv` to select the features used in the scatter plot.
4. Run `python pair_plot.py` to see the required pair plot.
