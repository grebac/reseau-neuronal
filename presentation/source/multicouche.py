from .perceptron import Perceptron, identityActivation, sigmoidActivation, tanhActivation
from .monocouche_matrice import showPlotlyColored
import pandas as pd 
import numpy as np
import os
import plotly.graph_objects as go

class reseauMultiCouche:
    def __init__(self, row_num=3, iterationMax=20000, EmoyenneMin=0.001, coucheCache=2, learningRate=0.8, activationFuncCache=sigmoidActivation(), activationFuncSortie=sigmoidActivation()):
        self.iterationMax = iterationMax
        self.EmoyenneMin = EmoyenneMin
        self.coucheCache = []
        for i in range(coucheCache):
            self.coucheCache.append(Perceptron(row_num, learningRate=learningRate, activationFunc=activationFuncCache))
        self.coucheSortie = Perceptron(coucheCache+1, learningRate=learningRate, activationFunc=activationFuncSortie)

    
    def processBoucle(self, inputs:np.array, labels:np.array):
        EMoyenne = 2000
        iteration = 0
        while EMoyenne > self.EmoyenneMin and iteration < self.iterationMax:
            EMoyenne = self.process(inputs, labels)
            iteration = iteration + 1
        # print("Le modele a ete resolu en {} itérations avec un Erreur moyenne de {}".format(iteration, EMoyenne))
        return (EMoyenne, iteration)
    
    def process(self, inputs:np.array, labels:np.array):
        activations_list = []

        # Prédiction couche cachée
        for i in range(len(self.coucheCache)):
            neurone = self.coucheCache[i]
            
            # Yc
            Yk = neurone.calculateYk(inputs)
            
            # On l'ajoute à la liste (création de l'input de la couche suivante)
            activations_list.append(Yk)

        # Convert the list of activations to a numpy array and reshape it to have one activation per column
        YcList = np.transpose(np.array(activations_list))

        # On ajoute un colonne de valeurs virtuelles 1 
        YcList = np.insert(YcList, 0, 1, axis=1)

        

        
        # Couche de sortie (mêmes calculs que pour la couche cachée)
        Zs = self.coucheSortie.calculateYk(YcList)
        
        
        # Calcul de l'erreur moyenne
        EMoyenne = self.coucheSortie.calculateMeanError(YcList, labels)
        print("EMoyenne : {}".format(EMoyenne))

        # Calcul du delta + retro-propagation
        errorSortie = self.coucheSortie.calculateError(labels, Zs)
        deltaSortie = self.coucheSortie.calculateDelta(errorSortie, Zs, YcList)        
        
        
        signalErrorSortie = self.coucheSortie.activationFunc.phi(Zs) * errorSortie
        # On passe à la couche cachée
        # Ec = Phi'(Pc) * Phi(Ps) * Wcs
        for i in range(len(self.coucheCache)):
            neuroneCache:Perceptron = self.coucheCache[i]
            
            # Pour chaque neurone, on doit calculer la somme pondérée des erreurs qu'il a engendré dans les neurones de la prochaine couche
            # On a besoin de créer une liste des poids de la prochaine couche lié à notre neurone
            # Ici on n'a qu'un neurone de sortie donc c'est facile
            index = i+1
            poids = np.array(self.coucheSortie.poids[index])
            sommePondéréErreurs = signalErrorSortie * poids
                        
            deltaCache = neuroneCache.calculateDelta(sommePondéréErreurs, YcList[:,index], inputs)
            neuroneCache.applyDelta(deltaCache)
        
        # On applique la modification de poids de la couche sortie après avoir traité toutes les couches cachées
        self.coucheSortie.applyDelta(deltaSortie)
        
        return EMoyenne
    
    def predictRegression(self, inputs:np.array):
        activations_list = []

        # Prédiction couche cachée
        for i in range(len(self.coucheCache)):
            neurone = self.coucheCache[i]
            
            # Yc
            Yk = neurone.calculateYk(inputs)
            
            # On l'ajoute à la liste (création de l'input de la couche suivante)
            activations_list.append(Yk)

        # Convert the list of activations to a numpy array and reshape it to have one activation per column
        YcList = np.transpose(np.array(activations_list))

        # On ajoute un colonne de valeurs virtuelles 1 
        YcList = np.insert(YcList, 0, 1, axis=1)
        
        # Couche de sortie (mêmes calculs que pour la couche cachée)
        Zs = self.coucheSortie.calculateYk(YcList)

        return Zs
            

def importData417(excel_file_path):
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    excel_file_path = os.path.join(dir_path, excel_file_path)
    
    # Read the Excel file into a pandas DataFrame
    df = pd.read_csv(excel_file_path, header=None)
    numpy_array = df.values
    
    # On ne garde que la première colonne
    inputs:np.array = numpy_array[:, 0]
    # On ajoute un colonne de valeurs virtuelles 1 
    inputs = np.column_stack((np.ones(inputs.shape), inputs))
    
    # La deuxième colonne est le label
    # Numpy s'occupe de transposer le vecteur ligne en vecteur colonne (nécessaire pour le bon fonctionnement de la multiplication matricielle)
    labels = numpy_array[:,1]

    row_num = inputs.shape[1]
    output_num = 1

    return inputs, labels, row_num, output_num        
        
def importData414(excel_file_path):
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

    row_num = inputs.shape[1]
    output_num = 3

    return inputs, labels, row_num, output_num

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
    
    showPlotlyColored2([model.coucheCache[0].getPlotlySeuillage("Neurone cache 1"), model.coucheCache[1].getPlotlySeuillage("Neurone cache 2")], np.column_stack((labels, inputs[:, 1:])))


def main412(coucheCache=3):
    print('main412()')
    inputs, labels, row_num, output_num = importData('Datas\\table_4_12.csv')
    
    model = reseauMultiCouche(coucheCache=coucheCache)
    model.processBoucle(inputs, labels)
    
    # Create a list to store the results of getPlotlySeuillage for each coucheCache
    plotly_data = []

    # Loop through each coucheCache in the model
    for i in range(len(model.coucheCache)):
        # Get the plotly data for this coucheCache
        plotly_data.append(model.coucheCache[i].getPlotlySeuillage(f"Neurone cache {i+1}"))

    # Prepare the labels and inputs for the plot
    plot_labels_inputs = np.column_stack((labels, inputs[:, 1:]))

    # Show the plot
    showPlotlyColored2(plotly_data, plot_labels_inputs)

def main414(coucheCache=3):
    print('main414()')
    inputs, labels, row_num, output_num = importData414('Datas\\table_4_14.csv')

    # On crée un réseau neuronal monocouche avec autant de neurones qu'il nous faut de sorties
    multicouches = list[reseauMultiCouche]()
    for i in range(output_num):
        multicouches.append(reseauMultiCouche(coucheCache=coucheCache))
        (E, iteration) = multicouches[i].processBoucle(inputs, labels[:,i])
        print("Erreur moyenne : {} en {} itérations".format(E, iteration))


    for j in range(len(multicouches)):
        # Create a list to store the results of getPlotlySeuillage for each coucheCache
        plotly_data = []
    
        # Loop through each coucheCache in the model
        for i in range(len(multicouches[j].coucheCache)):
            # Get the plotly data for this coucheCache
            plotly_data.append(multicouches[j].coucheCache[i].getPlotlySeuillage(f"Neurone cache {i+1}"))

        # Prepare the labels and inputs for the plot
        plot_labels_inputs = np.column_stack((labels[:,j], inputs[:, 1:]))

        # Show the plot
        showPlotlyColored2(plotly_data, plot_labels_inputs, f"Classe {j+1}")
    

def main417(coucheCache=3, iterationMax=20000, learningRate=0.8):
    inputs, labels, row_num, output_num = importData417('Datas\\table_4_17.csv')
    
    model = reseauMultiCouche(coucheCache=coucheCache, row_num=row_num, iterationMax=iterationMax, learningRate=learningRate, activationFuncCache=sigmoidActivation(), activationFuncSortie=identityActivation())
    (EMoyenne, iteration) = model.processBoucle(inputs, labels)
    print("Le modele a ete resolu en {} itérations avec un Erreur moyenne de {}".format(iteration, EMoyenne))
    
    # Create a list to store the results of getPlotlySeuillage for each coucheCache
    plotly_data = []
    plotly_data = np.column_stack((inputs[:,1], model.predictRegression(inputs)))

    # Prepare the labels and inputs for the plot
    plot_labels_inputs = np.column_stack((inputs[:, 1:], labels))

    # Show the plot
    showPlotlyRegression(plot_labels_inputs, plotly_data)

def showPlotlyColored2(lignesDecision, data, title=''):
    x_data = []
    y_data = []
    color_data = []
    
    for d in data:
        x_data.append(d[1])
        y_data.append(d[2])
        color_data.append('green' if d[0] > 0 else 'red')
    
    # Créer un scatter plot pour les points
    scatter_trace = go.Scatter(x=x_data, y=y_data, mode='markers', name='Values',
                               marker=dict(color=color_data))

    # Créer la figure
    fig = go.Figure()

    # Ajouter les traces à la figure
    fig.add_trace(scatter_trace)
    for ligneDecision in lignesDecision:
        fig.add_trace(ligneDecision)

    # Calculate the range for x and y axis
    x_range = [min(x_data) - 1, max(x_data) + 1]
    y_range = [min(y_data) - 1, max(y_data) + 1]

    # Ajouter une mise en page
    fig.update_layout(title=title,
                    xaxis=dict(title='X-Axis', range=x_range),
                    yaxis=dict(title='Y-Axis', range=y_range))

    # Afficher la figure
    fig.show()
    
def showPlotlyRegression(dots_to_predict: np.ndarray, predicted_dots: np.ndarray, title = 'Regression'):
    # Create a scatter plot of the dots to predict
    dots_to_predict_trace = go.Scatter(
        x=dots_to_predict[:, 0],  # X values
        y=dots_to_predict[:, 1],  # Y values
        mode='markers',
        name='Dots to Predict',
        marker=dict(color='red')  # Color the dots red
    )

    # Create a scatter plot of the predicted dots
    predicted_dots_trace = go.Scatter(
        x=predicted_dots[:, 0],  # X values
        y=predicted_dots[:, 1],  # Y values
        mode='markers',
        name='Predicted Dots',
        marker=dict(color='blue')  # Color the dots blue
    )

    # Create the layout for the plot
    layout = go.Layout(
        title='Regression',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
    )

    # Create the figure and add the two scatter plots
    fig = go.Figure(data=[dots_to_predict_trace, predicted_dots_trace], layout=layout)
    fig.update_layout(title=title)
    
    # Show the figure
    fig.show()

if __name__ == '__main__':
    # XOR()
    # main412()
    main414()
    
