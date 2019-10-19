# Xabier Etxezarreta
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate, ParameterGrid, GridSearchCV

class CrossValidator:
    def __init__(self, dataset, k_folds, algorithm):
        self.dataset = shuffle(dataset, random_state=0)
        self.k_folds = k_folds
        self.algorithm = algorithm
        # Dividimos el dataset en folds iguales
        self.folds = np.split(self.dataset, k_folds) # Lista de bloques (dataframes)
    
    def getFolds(self, k):
        train = []
        test = None
        for i, fold in enumerate(self.folds):
            if k == i:
                test = fold
            else:
                train.append(fold)
        return pd.concat(train), test

    def crossValidate(self):
        score = 0
        for i in range(self.k_folds):
            train, test = self.getFolds(i)
            # Entrenamos el modelo con el dataframe de train
            x_train = train.iloc[:, :-1]
            y_train = train.iloc[:,-1]
            model = self.algorithm.fit(x_train, y_train)

            # Hacemos predicciones con el dataframe de test para sacar
            # el accuracy
            x_test = test.iloc[:, :-1]
            y_test = test.iloc[:,-1]
            y_pred = model.predict(x_test)

            # Calculamos el accuracy
            score += accuracy_score(y_test, y_pred)
        score /= self.k_folds
        return score     

# Preparación de los datos
iris = datasets.load_iris()
x = iris['data']
y = iris['target']

df = pd.DataFrame(x, columns=iris['feature_names'])
df['target'] = pd.Categorical.from_codes(y, iris['target_names'])
k_folds = 10
randomForest = RandomForestClassifier(random_state=0)

# Mi cross-validation
my_cv = CrossValidator(df, k_folds, randomForest)
score = my_cv.crossValidate()
print('My CV accuracy: ' + str(score*100) + '%')

# Sklearn cross-validation
cv = cross_validate(randomForest, x, y, cv=k_folds)
score = np.sum(cv["test_score"]/k_folds)
print('Sklearn CV accuracy: ' + str(score*100) + '%')

# RandomForest grid-search
all_params = {"n_estimators": [50, 100, 150], "max_depth": [1,2,3], "random_state":[0,1,2]}    

def rfGridSearch(algorithm, data):
    bestParams = None
    bestScore = 0    

    # Podemos añadir mas valores o parámatros al diccionario (tarda mas por el aumento de combinaciones)    
    grid = ParameterGrid(all_params)

    for params in grid:
        rf = algorithm(**params)
        my_cv = CrossValidator(data, k_folds, rf)
        score = my_cv.crossValidate()

        if score > bestScore:
            bestParams = params
            bestScore = score
    
    print("My GridSearch best score: " + str(bestScore*100) + '%')
    print("My GridSearch best params: " + str(bestParams))

# My grid-search
rfGridSearch(RandomForestClassifier, df)

# Sklearn grid-search
clf = GridSearchCV(RandomForestClassifier(), all_params, cv=k_folds)
clf.fit(x, y)
print("Sklearn GridSearch best score: " + str(clf.best_score_*100) + '%')
print("Sklearn GridSearch best params: " + str(clf.best_params_))
