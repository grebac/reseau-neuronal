from matrice import PerceptronMultiCouche
from monocouche_matrice import showPlotlyColored
import pandas as pd 
import numpy as np
import os
import plotly.graph_objects as go

class reseauMultiCouche:
    def __init__(self, row_num=3):
        self.coucheCache = []
        for i in range(2):
            self.coucheCache.append(PerceptronMultiCouche(row_num))
        self.coucheSortie = PerceptronMultiCouche(row_num)
        
        self.row_num = row_num
    
    def processBoucle(self, inputs:np.array, labels:np.array):
        EMoyenne = 1000
        iteration = 0
        while EMoyenne > 0.001 and iteration < 1000:
            EMoyenne = self.process(inputs, labels)
            iteration = iteration + 1
        print("Le modele a ete resolu en {} itérations avec un Erreur moyenne de {}".format(iteration, EMoyenne))
    
    def process(self, inputs:np.array, labels:np.array):
        activations_list = []

        # Prédiction couche cachée
        for i in range(len(self.coucheCache)):
            neurone = self.coucheCache[i]
            # Prédiction
            Kc = neurone.calculateY(inputs)
            
            # Activation Sigmoid
            Yc = neurone.activation(Kc)
            
            # On l'ajoute à la liste (création de l'input de la couche suivante)
            activations_list.append([Yc])

        # Convert the list of activations to a numpy array and reshape it to have one activation per column
        YcList = np.array(activations_list).reshape(4, 2)

        # On ajoute un colonne de valeurs virtuelles 1 
        YcList = np.insert(YcList, 0, 1, axis=1)

        

        
        # Couche de sortie (mêmes calculs que pour la couche cachée)
        Ps = self.coucheSortie.calculateY(YcList).flatten()
        Zs = self.coucheSortie.activation(Ps)
        
        
        # Calcul de l'erreur + Emoyenne
        errorSortie = self.coucheSortie.calculateError(Zs, np.subtract(labels, Zs.flatten()))        
        EMoyenne = self.coucheSortie.calculateEmoyen(errorSortie)
        print("EMoyenne : {}".format(EMoyenne))

        # Retro-propagation de l'erreur de sortie + mise à jour
        deltaSortie = self.coucheSortie.calculateDelta(YcList, errorSortie)
        self.coucheSortie.miseAjourPoids(deltaSortie)
        
        # On passe à la couche cachée
        # Ec = Phi'(Pc) * Phi(Ps) * Wcs
        for i in range(len(self.coucheCache)):
            neuroneCache:PerceptronMultiCouche = self.coucheCache[i]
            
            # Pour chaque neurone, on doit calculer la somme pondérée des erreurs qu'il a engendré dans les neurones de la prochaine couche
            # On a besoin de créer une liste des poids de la prochaine couche lié à notre neurone
            # Ici on n'a qu'un neurone de sortie donc c'est facile
            index = i+1
            poids = np.array(self.coucheSortie.poids[index])
            sommePondéréErreurs = errorSortie * poids
            
            errorCache = neuroneCache.calculateError(YcList[:,index], sommePondéréErreurs)
            
            deltaCache = neuroneCache.calculateDelta(inputs, errorCache)
            neuroneCache.miseAjourPoids(deltaCache)
        
        return EMoyenne
            
        
        


def importData(excel_file_path):
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    excel_file_path = os.path.join(dir_path, excel_file_path)
    
    # Read the Excel file into a pandas DataFrame
    df = pd.read_csv(excel_file_path, header=None)
    numpy_array = df.values
    
    # On ne garde que les deux premières colonnes
    inputs:np.array = numpy_array[:, :2]
    # On ajoute un colonne de valeurs virtuelles 1 
    inputs = np.insert(inputs, 0, 1, axis=1)
    
    # Les autres colonnes sont des labels
    # Numpy s'occupe de transposer le vecteur ligne en vecteur colonne (nécessaire pour le bon fonctionnement de la multiplication matricielle)
    labels = numpy_array[:,2:]
    
    labels = labels[:,0]


    row_num = inputs.shape[1]
    output_num = 1

    return inputs, labels, row_num, output_num

def XOR():
    inputs = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    labels = np.array([0, 1, 1, 0])
    
    row_num = inputs.shape[1]
    output_num = 1

    model = reseauMultiCouche()
    model.processBoucle(inputs, labels)
    
    showPlotlyColored(model.coucheCache[0].getPlotlySeuillage("Not trained"), np.column_stack((labels, inputs[:, 1:])))
    showPlotlyColored(model.coucheCache[1].getPlotlySeuillage("Not trained"), np.column_stack((labels, inputs[:, 1:])))


def main412():
    print('main412()')
    inputs, labels, row_num, output_num = importData('Datas\\table_4_12.csv')
    
    model = reseauMultiCouche()
    
    showPlotlyColored(model.getPlotlySeuillage("Not trained"), np.column_stack((labels, inputs[:, 1:])))

if __name__ == '__main__':
    XOR()
    # main412()
    
