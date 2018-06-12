
# NBA Win Prediction Model


```python
import pandas as pd
import numpy as np
```

This model will include a dataset that is available on Fivethirtyeights github page. The code below loads that data and then prints off the first couple of rows from the dataset.


```python
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
data = pd.read_csv(url, sep=',')
print(data.head())
```

The follwing code aims to filter only the data that could be used to predict regular season NBA games. For the model I only include data from seasons starting in 2007, I eliminate all of the copies of games, and finally include only games that took place during the regular season. With the data filtered, the goal is now to only include variables that are available before a regualar season game begins. 


```python
df = data.query('year_id >= 2007 & _iscopy == 0 & is_playoffs == 0')
```

This model includes a teams points in the previous game (t-1), the amount of points that team allowed in the previous game (t-1), a teams elo rating entering the game, their opponets elo rating entering the game, and finally whether the game was home, away, or neutral. The target variable for the model will be if the team one or lost. 


```python
NBA = df[['pts','opp_pts', 'elo_i', 'opp_elo_i', 'game_location', 'game_result' ]]
```


```python
NBA.loc["pts"] = NBA[['pts']].shift(1)
NBA.loc['opp_pts'] = NBA[['opp_pts']].shift(1)
```


```python
NBA = NBA.dropna(axis = 0)
```


```python
NBA = np.random.shuffle(NBA)
```


```python
x = NBA.drop(['game_result'], axis = 1)
X = pd.get_dummies(x)
Y = NBA[['game_result']]
```

Next we will create a test train split in order to train on and test the model on different sets of data. This helps to not overfit the model to the training data.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 123)
```


```python
from xgboost import XGBClassifier
model = XGBClassifier()
```

In order to find the best XGBoost model we will use SKLearn's GridSearchCV funtion. Below are links to the documentation for XGBoost parameters and SKLearn's GridSearchCV function. XGBoost:http://xgboost.readthedocs.io/en/latest/parameter.html
GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html


```python
n_estimators = [100,150,200,250,300]
max_depth = [2,4,6,8,10]
min_child_weight = [1,2,3,4,5]
param_grid = dict(n_estimators = n_estimators,max_depth = max_depth, min_child_weight = min_child_weight)
```


```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(model, param_grid, scoring="accuracy", cv = 5, n_jobs=2)
grid_result = grid_search.fit(X_train, y_train)
```


```python
print("%f %s" % (grid_result.best_score_, grid_result.best_params_))
```


```python
WinModel = XGBClassifier(n_estimators = 100, min_child_weight = 1, max_depth = 2)
WinModel.fit(X_train, y_train)
```


```python
y_pred = pd.DataFrame(WinModel.predict(X_test))
print(y_pred.head())
```


```python
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
```

Great! Now we know the accuracy of the model, but what is we want to express our predictions in the form of a probability? using predict_proba will take care of that.


```python
y_preds_percent = pd.DataFrame(WinModel.predict_proba(X_test))
print(y_preds_percent.head())
```

If you have any further questions about the code or model above feel free to reach out to me on twitter @kendallasmith.
