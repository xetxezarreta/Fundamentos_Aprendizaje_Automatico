import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

class CrossValidator:
    def __init__(self, dataset, k_folds, algorithm):
        self.dataset = shuffle(dataset)
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

            score += accuracy_score(y_test, y_pred)
        score = score / self.k_folds
        return score
        
        

iris = datasets.load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['target'] = pd.Categorical.from_codes(iris['target'], iris['target_names'])

cv = CrossValidator(df, 10, RandomForestClassifier())
score = cv.crossValidate()
print('Accuracy: ' + str(score*100) + '%')
