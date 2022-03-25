import pandas as pd
import sklearn as sk
import numpy as np
from sklearn import ensemble, model_selection, metrics, neural_network, preprocessing

X = pd.read_csv('~/data/MLProject/xTrainFinal.csv').set_index('sample_id')
Y = pd.read_csv('~/data/MLProject/y_train.csv').set_index('sample_id')
test = pd.read_csv('~/data/MLProject/xTestFinal.csv')
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size=.1)
reg = ensemble.HistGradientBoostingRegressor(verbose=2, max_leaf_nodes=None, max_iter=200, min_samples_leaf=10, random_state=1000, validation_fraction=.1, tol=1e-9)

# reg.fit(X, Y)
# res = pd.DataFrame({'target':reg.predict(test.drop(columns='sample_id'))}, index=test['sample_id'])
# print(res)
# pd.DataFrame(res).to_csv("~/data/MLProject/HGB_median.csv")

reg.fit(xTrain, yTrain)
print(int(np.sqrt(metrics.mean_squared_error(yTest, reg.predict(xTest))) * 10000))