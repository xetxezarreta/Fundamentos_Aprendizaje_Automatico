# Implementación del algoritmo KNN en Python

import csv
import math
import random
import operator
from math import *
from decimal import Decimal
import numpy as np
from scipy.spatial import distance


def loadDataset(filename, split, trainingSet=[], testSet=[]):

    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

class customKNN:
    def __init__(self, k, distance, training_set, test_set):
        self.__distance = distance
        self.__k = k
        self.__trainingSet = training_set
        self.__testSet = test_set

    def euclideanDistance(self, instance1, instance2, length):
        """
        Para poder hacer predicciones necesitamos calcular la distancia entre instancias.
        Esto se necesita para localizar las K instancias más similares en el conjunto de entrenamiento para un miembro dado del dataset de test.
        Recordad que para calcular la distancia, necesitamos que las instancias sean numéricas.
        Además, tendremos que controlar que campos incluir en el cálculo de la distancia. Para eso utilizaremos el campo length.

        :param instance1
        :param instance2
        :param length: una forma de limitar los campos que se usarán en el cálculo de la distancia es elegir los primeros x campos
        :return: distancia euclidea entre las dos instancias
        """
        distance = 0
        for x in range(length):
            #implementar cálculo de la distancia euclidea
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def manhattanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            #implementar cálculo de la distancia manhattan
            distance += abs(instance1[x] - instance2[x])
        return distance

    def mahalanobisDistance(self, instance1, instance2, length):

        dist = 0
        #cálculo de la distancia mahalanobis. Solo tenéis que utilizar la librería de Numpy para implementarla
        #se diferencia de la distancia euclídea en que tiene en cuenta la correlación entre las variables aleatorias. 
        z = zip(instance1, instance2)
        cov = np.cov(z) # matriz de covarianza
        inv = np.linalg.inv(cov) # inversa de matriz de covarianza

         
        return dist

    def getDistance(self, instance1, instance2, length):
        if (self.__distance == "euclidean"):
            return self.euclideanDistance(instance1, instance2, length)
        elif (self.__distance == "manhattan"):
            return self.manhattanDistance(instance1, instance2, length)
        elif (self.__distance == "mahalanobis"):
            return self.mahalanobisDistance(instance1, instance2, length)
        else:
            return 0

    def getNeighbors(self, testInstance):
        """
        Seleccionar las k instancias más parecidas basándose en una instancia no vista anteriormente.
        :param testInstance
        :return: lista de vecinos
        """
        distances = []
        length = len(testInstance) - 1
        for x in range(len(self.__trainingSet)):
            dist = self.getDistance(testInstance, self.__trainingSet[x], length)
            distances.append((self.__trainingSet[x], dist))

        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(self.__k):
            neighbors.append(distances[x][0])
        return neighbors


    def getResponse(self, neighbors):
        """
        Partiendo de una lista de vecinos, hay que predecir su clase.
        Una forma para hacerlo es que cada vecino vote por su atributo clase, y se cogerá el voto de la mayoría como predicción.
        Se asume que el último atributo es el atributo clase para cada vecino.

        :param neighbors:
        :return: clase a la que pertenece la instancia
        """
        classVotes = {}
        for x in range(len(neighbors)):
            #hacer un recuento de votos por cada vecino
            print("")

        return 1#clase a la que pertenece la instancia





    def getAccuracy(self, predictions):
        correct = 0
        for x in range(len(self.__testSet)):
            if self.__testSet[x][-1] == predictions[x]:
                correct += 1
        return (correct / float(len(self.__testSet))) * 100.0

    def predict(self):
        predictions = []
        for x in range(len(self.__testSet)):
            neighbors = self.getNeighbors(self.__testSet[x])
            result = self.getResponse(neighbors)
            predictions.append(result)
            print('> predicted=' + repr(result) + ', actual=' + repr(self.__testSet[x][-1]))
        accuracy = self.getAccuracy(predictions)
        print('Accuracy: ' + repr(accuracy) + '%')

def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('07_ejercicios_estadistica_1/data/iris.data', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))

    k = 3

    knn = customKNN(k, "manhattan", trainingSet, testSet)
    knn.predict()

main()

# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/