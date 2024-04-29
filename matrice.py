import numpy as np
import random
from numpy.core.multiarray import array as array
import plotly.graph_objects as go
import os
import pandas as pd

class Perceptron:
    def __init__(self, row_num, IterationMax=100, EMoyenneMax=0.125, sigma=0.1, tauxApprentissage=0.2, sigmoidCoef = 1, debug=False):
        self.poids = np.array([])
        for i in range(row_num):
            self.poids = np.append(self.poids, random.gauss(0, sigma=sigma))

        self.IterationMax = IterationMax
        self.ErrorMoyenMax = EMoyenneMax
        self.tauxApprentissage = tauxApprentissage
        self.SigmoidCoef = sigmoidCoef
        self.debug = debug
        
    # Le résultat est une colonne de X lignes (avec X le nombre d'exemples dans le dataset) avec la prédiction du modèle "Y"
    def calculateY(self, inputs:np.array):
        return np.dot(inputs,self.poids)

    # On calcule l'erreur entre les labels et les valeurs prédites
    def calculateError(self, labels:np.array, Y:np.array):
        return labels - Y

    # On obtient pour chaque poids un delta
    def calculateDelta(self, inputs, errors):
        return self.tauxApprentissage * np.dot(errors, inputs)
    
    def batchProcess(self, inputs:np.array, labels:np.array, lots:int):
        EMoyenne = 0
        iterationTotal = 0
        for i in range(0, inputs.shape[0], lots):
            (EMoyenne, iteration) = self.process(inputs[i:i+lots], labels[i:i+lots])
            iterationTotal += iteration
        return (EMoyenne, iterationTotal)
        
    def activation(self, Y:np.array):
        return np.where(Y > 0, 1, -1)
    
    def miseAjourPoids(self, deltaError:np.array):
        self.poids = self.poids + deltaError
        
    # Em = SUM[ (Ds - Ps)² / 2 * N ]
    def calculateEmoyen(self, Errors:np.array, N:int):
        return np.sum(Errors**2) / (N * 2)  

    def process(self, inputs:np.array, labels:np.array):
        EMoyenne = 1    
        iteration = 1
        while EMoyenne > self.ErrorMoyenMax and iteration < self.IterationMax:
            # On calcule la somme pondérée des entrées
            arrayY:np.array = self.calculateY(inputs)
            
            # On applique la fonction d'activation (seuillage ou sigmoid)
            arrayActivation:np.array = self.activation(arrayY)
        
            # On calcule l'erreur (différence entre les labels et les valeurs prédites)
            arrayError:np.array = self.calculateError(labels, arrayY)

            # On calcule le changement à apporter aux poids            
            deltaArray:np.array = self.calculateDelta(inputs, arrayError)                
            
            # On applique le changement aux poids
            self.miseAjourPoids(deltaArray)
            
            # On calcule l'erreur moyenne
            EMoyenne = self.calculateEmoyen(arrayError, inputs.shape[0])

            if self.debug:
                print("Prediction: {}".format(arrayY))
                print("Seuillage : {}".format(arrayActivation))
                print("Errors : {}".format(arrayError))
                print("Delta : {}".format(deltaArray)) 
                print("Updated Poids : {}".format(self.poids))
                print("Erreur moyenne : {}".format(EMoyenne))
            
            iteration += 1
        
        if self.debug:
            print("Il a fallu {} itérations pour résoudre le modèle".format(iteration))
            print("Les poids sont : {}".format(self.poids))
            print("L'erreur moyenne est de : {}".format(EMoyenne))
        return (EMoyenne, iteration)
        
    def getPlotlySeuillage(self, name='Droite de seuillage'):
         # Calculate the slope and intercept of the decision boundary
        slope = - self.poids[1] / self.poids[2]
        intercept = - self.poids[0] / self.poids[2]
        
        # On a maintenant besoin d'élonger la droite (car elle ne se base que sur deux points)
        # On crée une série de valeurs X
        xSeuillage = np.linspace(-200, 200, 400)
        
        # On calcule pour chaque point son ordonnée
        ySeuillage =  slope * xSeuillage + intercept
        
        # On crée une droite de seuillage
        seuillage = go.Scatter(x=xSeuillage, y=ySeuillage, mode='lines', name=name)
        
        return seuillage
        
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
    
class PerceptronMultiCouche(Perceptron):
    # Voir tab 4.3 page 72 du cours
    def activation(self, Y:np.array):
        return 1 / (1 + np.exp(-Y * self.SigmoidCoef))
    
    # Voir tab 4.4 page 72 du cours
    # En bref : Zs * (1 - Zs)
    def deriveeSigmoid(self, Zs:np.array):
        return self.SigmoidCoef * Zs * (1 - Zs)
    
    # L'erreur dans le multi-couche : 
    # E = phi'(prédiction) * secondParameter
    # secondParameter : 
    # Si on est dans un neurone de sortie, "secondParameter" est juste égal à (ds - ys)
    # Si on ets dans un neurone caché, il vaut "la somme des erreurs des neurones de la prochaine couche, pondéré par le poid attribué au neurone acutel"
    def calculateError(self, Zs: np.array, secondParameter:np.array):
        derivePhi = self.deriveeSigmoid(Zs)
        return secondParameter * derivePhi.flatten()
    
    # delta = n * Errors * inputs 
    def calculateDelta(self, inputs, errors):
        return self.tauxApprentissage * np.dot(errors, inputs)
    
    # Em = 1/2 * SUM[ (Ds - Ps)² ]
    def calculateEmoyen(self, Errors:np.array, N:int=-1):
        return 1/2 * np.sum(Errors**2)

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

def processFile(filename):
    rawData = openFile(filename)
    (inputs, labels, row_num) = processRawData(rawData)
    print("Inputs : {}".format(inputs))
    print("Labels : {}".format(labels))
    
    
    
    perceptron = Perceptron(row_num, IterationMax=10000, EMoyenneMax=0.56, sigma=0, tauxApprentissage=0.00014, debug=False)
    (EMoyenne, iteration) = perceptron.process(inputs, labels)
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

if __name__ == "__main__":    
    processFile("Datas/table_2_11.csv")
    
    
    
    