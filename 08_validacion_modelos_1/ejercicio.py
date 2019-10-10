from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class CrossValidator():
    def __init__(self, dataset, k_folds, algorithm):
        self.dataset = dataset
        self.k_folds = k_folds
        self.algorithm = algorithm
    
    def getFolds(self):
        pass

    def crossValidate(self):
        pass

