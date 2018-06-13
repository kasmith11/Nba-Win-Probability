import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
data = pd.read_csv(url, sep=',')

df = data.query('year_id >= 2007 & _iscopy == 0 & is_playoffs == 0')

NBA = df[['pts','opp_pts', 'elo_i', 'opp_elo_i', 'game_location', 'game_result' ]]
NBA[["pts"]] = NBA[['pts']].shift(1)
NBA[['opp_pts']] = NBA[['opp_pts']].shift(1)
NBA = NBA.dropna(axis = 0)
x = NBA.drop(['game_result'], axis = 1)
X = pd.get_dummies(x)
Y = NBA[['game_result']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 123)

from xgboost import XGBClassifier
model = XGBClassifier()
n_estimators = [100,150,200,250,300]
max_depth = [2,4,6,8,10]
min_child_weight = [1,2,3,4,5]
param_grid = dict(n_estimators = n_estimators,max_depth = max_depth, min_child_weight = min_child_weight)

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(model, param_grid, scoring="accuracy", cv = 5, n_jobs=2)
grid_result = grid_search.fit(X_train, y_train)

print("%f %s" % (grid_result.best_score_, grid_result.best_params_))

WinModel = XGBClassifier(n_estimators = 100, min_child_weight = 1, max_depth = 2)
WinModel.fit(X_train, y_train)
y_pred = pd.DataFrame(WinModel.predict(X_test))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
