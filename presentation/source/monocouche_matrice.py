from .perceptron import Perceptron, sigmoidActivation
import pandas as pd 
import numpy as np
import os
import plotly.graph_objects as go

def importData31(excel_file_path):
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

def table31(IterationMax=1000, EMoyenneMax=0.01, sigma=0.1, tauxApprentissage=0.0005):
    inputs, labels, row_num, output_num = importData31('Datas\\table_3_1.csv')

    # On crée un réseau neuronal monocouche avec autant de neurones qu'il nous faut de sorties
    monocouche = list[Perceptron]()
    for i in range(output_num):
        monocouche.append(Perceptron(row_num, IterationMax=IterationMax, MeanErrorMax=EMoyenneMax, sigma=sigma, learningRate=tauxApprentissage))
        (E, iteration) = monocouche[i].processBatch(inputs, labels[i], 150)
        print("Erreur moyenne : {} en {} itérations".format(E, iteration))

    # Affichage des résultats indépendants
    showPlotlyColored(monocouche[0].getPlotlySeuillage("label 0"), np.column_stack((labels[0], inputs[:,1:]))) 
    showPlotlyColored(monocouche[1].getPlotlySeuillage("label 1"), np.column_stack((labels[1], inputs[:,1:]))) 
    showPlotlyColored(monocouche[2].getPlotlySeuillage("label 2"), np.column_stack((labels[2], inputs[:,1:]))) 
    
    return monocouche


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
    
    first_labels = labels[:,0]
    second_labels = labels[:,1]
    thrid_labels = labels[:,2]
    fourth_labels = labels[:,3]
    
    final_labels = np.array([first_labels, second_labels, thrid_labels, fourth_labels])


    row_num = inputs.shape[1]
    output_num = labels.shape[1]

    return inputs, final_labels, row_num, output_num
    
def table35(IterationMax=1000, EMoyenneMax=0.01, sigma=0.1, tauxApprentissage=0.0005):
    inputs, labels, row_num, output_num = importData35('Datas\\table_3_5.csv')

    # On crée un réseau neuronal monocouche avec autant de neurones qu'il nous faut de sorties
    monocouche = list[Perceptron]()
    for i in range(output_num):
        monocouche.append(Perceptron(row_num, IterationMax=IterationMax, MeanErrorMax=EMoyenneMax, sigma=sigma, learningRate=tauxApprentissage))
        (E, iteration) = monocouche[i].processBatch(inputs, labels[i], 4)
        print("Erreur moyenne : {} en {} itérations".format(E, iteration))
        
    return monocouche


def importSigne(excel_file_path):
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    excel_file_path = os.path.join(dir_path, excel_file_path)
    
    # Read the Excel file into a pandas DataFrame
    df = pd.read_csv(excel_file_path, header=None)
    numpy_array = df.values
    

    # On filtre un trainingDataset et un validationDataset
    (trainingDataset, validationDataset, indexesValidation) = splitDatasets(numpy_array)
    
    training_inputs = trainingDataset[:, :-5]
    training_inputs = np.insert(training_inputs, 0, 1, axis=1)
    
    
    
    # Les autres colonnes sont des labels
    # Numpy s'occupe de transposer le vecteur ligne en vecteur colonne (nécessaire pour le bon fonctionnement de la multiplication matricielle)
    training_labels = trainingDataset[:,-5:]
    
    first_labels = training_labels[:,0]
    second_labels = training_labels[:,1]
    thrid_labels = training_labels[:,2]
    fourth_labels = training_labels[:,3]
    fifth_labels = training_labels[:,4]
    
    training_final_labels = np.array([first_labels, second_labels, thrid_labels, fourth_labels, fifth_labels])

    # On merge les inputs et les labels en un dataset
    trainingDataset = [training_inputs, training_final_labels]



    # On récupère jusqu'au 4 dernières colonnes
    validation_inputs:np.array = validationDataset[:, :-5]
    # On ajoute un colonne de valeurs virtuelles 1 
    validation_inputs = np.insert(validation_inputs, 0, 1, axis=1)

    # Les autres colonnes sont des labels
    # Numpy s'occupe de transposer le vecteur ligne en vecteur colonne (nécessaire pour le bon fonctionnement de la multiplication matricielle)
    validation_labels = validationDataset[:,-5:]
    
    first_labels = validation_labels[:,0]
    second_labels = validation_labels[:,1]
    thrid_labels = validation_labels[:,2]
    fourth_labels = validation_labels[:,3]
    fifth_labels = validation_labels[:,4]
    
    validation_final_labels = np.array([first_labels, second_labels, thrid_labels, fourth_labels, fifth_labels])

    validationDataset = [validation_inputs, validation_final_labels]

    row_num = training_inputs.shape[1]
    output_num = training_labels.shape[1]

    return trainingDataset, validationDataset, indexesValidation, row_num, output_num
        
def signe(trainingDataset:np.array, row_num, output_num, maxIteration=5000, learningRate=0.0005):
    inputs = trainingDataset[0]
    labels = trainingDataset[1]

    # On crée un réseau neuronal monocouche avec autant de neurones qu'il nous faut de sorties
    monocouche = list[Perceptron]()
    for i in range(output_num):
        monocouche.append(Perceptron(row_num, IterationMax=maxIteration, MeanErrorMax=0.00001, sigma=0.1, learningRate=learningRate, activationFunc=sigmoidActivation()))
        (E, iteration) = monocouche[i].processBatch(inputs, labels[i],250)
        print("Erreur moyenne : {} en {} itérations".format(E, iteration))
        
    return monocouche

def splitDatasets(numpy_array):    
    # 47 car 21 paramètres en XY => 21*2 = 42 + les 5 labels => 42+5 = 47
    # (+1 pour sauvegarder l'index de chaque élément) => 47 + 1 = 48
    trainingSet = np.empty((0,48))
    validationSet = np.empty((0,48))
    
    # On veut "keep track" de l'index de chaque élément
    allIndexes = np.arange(numpy_array.shape[0])
    numpy_array = np.column_stack((numpy_array, allIndexes))

    # On veut un jeu de données d'entrainement et de validation.
    # Attention, il nous faut répartir pour chaque classe une portion proportionnelle de chaque classe
    for classes in range(1,6):
        # On récupère les éléments classe par classe
        matchCurrentClass = numpy_array[:,-(6-classes)].astype(int)
        elements = numpy_array[matchCurrentClass == 1]
        
        # Pour chaque classe, on garde 1/6 des éléments 
        # 60 éléments par classes => 50 training + 10 validation
        np.random.shuffle(elements)
        trainingSet = np.vstack((trainingSet, elements[:50]))
        validationSet = np.vstack((validationSet, elements[50:]))

        # On mélange une dernière fois
        np.random.shuffle(validationSet)
        np.random.shuffle(trainingSet)
        
    return (trainingSet[:,:-1], validationSet[:,:-1], validationSet[:,-1])
      
        
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
     

def predict(monocouche:list[Perceptron], input:np.array):
    result = []
    classN = 1
    for neurone in monocouche:
        Yk = neurone.calculateYk(input)
        result.append(Yk)
    return result
 
    
if __name__ == "__main__":
    # monocouche = table31()
    # result = predict(monocouche, np.array((1,2.5, 1.5)))
    # print(result)
    
    # monocouche = table35()
    # result = predict(monocouche, np.array((0,0,1,0,0,0,0,1,0,0,0,1,1,0,1,1,0,0,1,0,0,0,0,1,0,0)))
    # print(result)
    
    trainingDataset, validationDataset, row_num, output_num = importSigne('Datas\\LangageDesSignes\\data_formatted.csv')
    monocouche = signe(trainingDataset, row_num, output_num)
    result = predict(monocouche, validationDataset[0][0])
    print("On obtient : {0} || On était censé obtenir : {1}".format(result, validationDataset[1][:,0]))