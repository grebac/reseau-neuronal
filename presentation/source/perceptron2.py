import numpy as np
import random
from numpy.core.multiarray import array as array
import plotly.graph_objects as go
import os
import pandas as pd
            
class activationFunction:
    def __init__(self) -> None:
        pass
    def activation(self, Y:np.array):
        pass
    def phi(self, Y:np.array):
        pass
    
class identityActivation(activationFunction):
    def __init__(self) -> None:
        pass
    
    def activation(self, Y:np.array):
        return Y
    
    def phi(self, Y:np.array):
        return 1
    
class sigmoidActivation(activationFunction):
    def __init__(self, coefSigmoid=1) -> None:
        self.coefSigmoid = coefSigmoid
    
    def activation(self, Y:np.array):
        return 1 / (1 + np.exp(-Y))
    
    def phi(self, Y:np.array):
        return Y * (1 - Y)
    
class Perceptron2:
    def __init__(self, row_num, IterationMax=100, MeanErrorMax=0.125, sigma=0.1, learningRate=0.2, debug=False, activationFunc=identityActivation(), isAdaline=False):
        self.initializeWeights(row_num, sigma)
        
        self.activationFunc:activationFunction = activationFunc
        self.learningRate = learningRate
        self.iterationMax = IterationMax
        self.MeanErrorMax = MeanErrorMax
        self.isAdaline = isAdaline

    def initializeWeights(self, row_num, sigma):
        self.poids = np.array([])
        for i in range(row_num):
            self.poids = np.append(self.poids, random.gauss(0, sigma=sigma))
            
    # X11 X12 X13 * [W0 W1 W2] =    P1
    # X21 X22 X23                   P2
    # X31 X32 X33                   P3
    # X41 X42 X43                   P4
    def __calculateP(self, inputs:np.array) -> np.array:
        return np.dot(inputs, self.poids)
    
    def calculateYk(self, inputs:np.array) -> np.array:
        P = self.__calculateP(inputs)
        return self.activationFunc.activation(P)
    
    def __calculateError(self, labels:np.array, Y:np.array) -> np.array:
        return labels - Y
    
    # Delta = n * (Ds - Ps) * inputs * phi(Ps)
    def calculateDelta(self, labels:np.arange, Y:np.array, inputs:np.array) -> np.array:
        errors = self.__calculateError(labels, Y)
        phi = self.activationFunc.phi(Y)
        
        tmp = np.dot(errors * phi, inputs)
        return self.learningRate * tmp 
    
    def calculateMeanError(self, inputs:np.array, labels:np.array) -> float:
        Yk = self.calculateYk(inputs)
        errors = self.__calculateError(labels, Yk)
        N = inputs.shape[0]
        
        return np.sum(errors**2) / (N * 2)
    
    def applyDelta(self, delta:np.array):
        self.poids = self.poids + delta
        
    def __process(self, inputs:np.array, labels:np.array):
        MeanError = 1
        iteration = 1
        while MeanError > self.MeanErrorMax and iteration < self.iterationMax:
            Yk = self.calculateYk(inputs)
            delta = self.calculateDelta(labels, Yk, inputs)
            self.applyDelta(delta)
            MeanError = self.calculateMeanError(inputs, labels)
            iteration += 1
        
        return (MeanError, iteration)
    
    def processBatch(self, inputs:np.array, labels:np.array, lots:int):
        MeanError = 0
        iterationTotal = 0
        for i in range(0, inputs.shape[0], lots):
            (MeanError, iteration) = self.__process(inputs[i:i+lots], labels[i:i+lots])
            iterationTotal += iteration
        
        return (MeanError, iterationTotal)
    
    def getPlotlyRegression(self, name='Droite de régression'):
        # Calculate the slope and intercept of the decision boundary
        slope = self.poids[1]
        intercept = self.poids[0]
        
        # On a maintenant besoin d'élonger la droite (car elle ne se base que sur deux points)
        # On crée une série de valeurs X
        xSeuillage = np.linspace(-200, 200, 400)
        
        # On calcule pour chaque point son ordonnée
        ySeuillage =  slope * xSeuillage + intercept
        
        # On crée une droite de seuillage
        seuillage = go.Scatter(x=xSeuillage, y=ySeuillage, mode='lines', name=name)
        
        return seuillage


def process2_11(filename):
    rawData = openFile(filename)
    (inputs, labels, row_num) = processRawData(rawData)
    print("Inputs : {}".format(inputs))
    print("Labels : {}".format(labels))
    
    
    
    perceptron = Perceptron2(row_num, IterationMax=10000, MeanErrorMax=0.56, sigma=0, learningRate=0.00014, debug=False, activationFunc=identityActivation())
    (EMoyenne, iteration) = perceptron.processBatch(inputs, labels, 30)
    print("Erreur moyenne : {}".format(EMoyenne))
    
    displayPlotly(np.column_stack([inputs[:,1], labels]), [perceptron.getPlotlyRegression()])
    
def openFile(filename):
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, filename)
    
    # Read data
    x1 = pd.read_csv(filename, header=None)
    datas = np.array(x1.values)
    
    return datas

def processRawData(rawData: np.array):
    labelColumn = rawData.shape[1] - 1
    
    # On ne garde que les deux premières colonnes
    inputs:np.array = rawData[:, :labelColumn]
    # On ajoute un colonne de valeurs virtuelles 1 
    inputs = np.insert(inputs, 0, 1, axis=1)
    
    # Les labels sont dans la dernière colonne
    # Numpy s'occupe de transposer le vecteur ligne en vecteur colonne (nécessaire pour le bon fonctionnement de la multiplication matricielle)
    labels = rawData[:,labelColumn]
    
    row_num = inputs.shape[1]
    
    return inputs, labels, row_num

def displayPlotly(datas:np.array, seuillages:list[go.Scatter]):    
    # On converti les données pour poltly
    x = datas[:,0]
    y = datas[:,1]

    # Create a scatter plot
    graph = go.Figure()

    # # On converti les données pour poltly
    datas = go.Scatter(x=x, y=y, mode='markers', name='Data')

    # # Set plot title and labels
    graph.add_trace(datas)

    # Add the seuillages    
    for s in seuillages:
        graph.add_trace(s)

    # # Stay zoomed on the datas
    margin_x = 0.1 * (np.max(x) - np.min(x))  # 10% of the x range
    margin_y = 0.1 * (np.max(y) - np.min(y))  # 10% of the y range
    graph.update_xaxes(range=[np.min(x) - margin_x, np.max(x) + margin_x])
    graph.update_yaxes(range=[np.min(y) - margin_y, np.max(y) + margin_y])
    
    # Show the plot
    graph.show()

if __name__ == "__main__":    
    process2_11("Datas/table_2_11.csv")

    
    