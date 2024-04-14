from matrice import Perceptron
import pandas as pd 
import numpy as np
import os
import plotly.graph_objects as go

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
    
    first_labels = labels[:,0]
    second_labels = labels[:,1]
    thrid_labels = labels[:,2]
    
    final_labels = np.array([first_labels, second_labels, thrid_labels])


    row_num = inputs.shape[1]
    output_num = labels.shape[1]

    return inputs, final_labels, row_num, output_num

def showPlotlyColored(ligneDecision, data):
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
    fig.add_trace(ligneDecision)

    # Calculate the range for x and y axis
    x_range = [min(x_data) - 1, max(x_data) + 1]
    y_range = [min(y_data) - 1, max(y_data) + 1]

    # Ajouter une mise en page
    fig.update_layout(title='Scatter Plot with Independent Line',
                    xaxis=dict(title='X-Axis', range=x_range),
                    yaxis=dict(title='Y-Axis', range=y_range))

    # Afficher la figure
    fig.show()

def table31():
    inputs, labels, row_num, output_num = importData('Datas\\table_3_1.csv')

    # On crée un réseau neuronal monocouche avec autant de neurones qu'il nous faut de sorties
    monocouche = list[Perceptron]()
    for i in range(output_num):
        monocouche.append(Perceptron(row_num, IterationMax=1000, EMoyenneMax=0.01, sigma=0.1, tauxApprentissage=0.0005))
        (E, iteration) = monocouche[i].process(inputs, labels[i])
        print("Erreur moyenne : {} en {} itérations".format(E, iteration))

    # Affichage des résultats indépendants
    showPlotlyColored(monocouche[0].getPlotlySeuillage("label 0"), np.column_stack((labels[0], inputs[:,1:]))) 
    showPlotlyColored(monocouche[1].getPlotlySeuillage("label 1"), np.column_stack((labels[1], inputs[:,1:]))) 
    showPlotlyColored(monocouche[2].getPlotlySeuillage("label 2"), np.column_stack((labels[2], inputs[:,1:]))) 
      
    # Affichage des résultats combinés
    # seuillages = []
    
    # seuillages.append(monocouche[0].getPlotlySeuillage("label 1"))
    # seuillages.append(monocouche[1].getPlotlySeuillage("label 2"))
    # seuillages.append(monocouche[2].getPlotlySeuillage("label 3"))
    
    # displayPlotly(inputs[:, 1:], seuillages)

def importData35(excel_file_path):
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    excel_file_path = os.path.join(dir_path, excel_file_path)
    
    # Read the Excel file into a pandas DataFrame
    df = pd.read_csv(excel_file_path, header=None)
    numpy_array = df.values
    
    # On récupère jusqu'au 4 dernières colonnes
    inputs:np.array = numpy_array[:, :-4]
    # On ajoute un colonne de valeurs virtuelles 1 
    inputs = np.insert(inputs, 0, 1, axis=1)
    
    # Les autres colonnes sont des labels
    # Numpy s'occupe de transposer le vecteur ligne en vecteur colonne (nécessaire pour le bon fonctionnement de la multiplication matricielle)
    labels = numpy_array[:,-4:]
    
    first_labels = labels[0,:]
    second_labels = labels[1,:]
    thrid_labels = labels[2,:]
    fourth_labels = labels[3,:]
    
    final_labels = np.array([first_labels, second_labels, thrid_labels, fourth_labels])


    row_num = inputs.shape[1]
    output_num = labels.shape[1]

    return inputs, final_labels, row_num, output_num
    
def table35():
    inputs, labels, row_num, output_num = importData35('Datas\\table_3_5.csv')

    # On crée un réseau neuronal monocouche avec autant de neurones qu'il nous faut de sorties
    monocouche = list[Perceptron]()
    for i in range(output_num):
        monocouche.append(Perceptron(row_num, IterationMax=1000, EMoyenneMax=0.01, sigma=0.1, tauxApprentissage=0.0005))
        (E, iteration) = monocouche[i].process(inputs, labels[i])
        print("Erreur moyenne : {} en {} itérations".format(E, iteration))
    
if __name__ == "__main__":
    table31()
    # table35()
        