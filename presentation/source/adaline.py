import random
import copy
import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np

class LabeledData:    
    def __init__(self, intput_list:list[float], label_list:list[float], fictional=True) -> None:        
        self.inputs = copy.deepcopy(intput_list)
        self.label = copy.deepcopy(label_list)
        self.currentLabel = self.label[0]
        if fictional:
            self.inputs.insert(0, 1.0)
    
    def __repr__(self) -> str:
        return f'{{label={self.label}, inputs={self.inputs}}}'
    
    def setCurrentLabel(self, labelIndex:int):
        self.currentLabel = self.label[labelIndex]

class Adaline:
    def __init__(self, learning_rate, inputs_number:int, sigma:float) -> None:
        self.learning_rate = learning_rate
        self.weights = []
        for index in range(inputs_number+1):
            self.weights.append(random.gauss(0, sigma=sigma))
    
    def __repr__(self):
        return f'Adaline: {self.learning_rate:}, {[float("%.6f" % weight) for weight in self.weights]}'

    def train(self, dataset:list[LabeledData], max_iteration:int, error_thresold:float):
        K = len(dataset)
        if len(dataset[0].inputs) != len(self.weights):
            print("Dataset unusable: inputs mismatch weights.")
            return -1
        
        iteration = 1
        mean_square_error = error_thresold + 1
        while iteration <= max_iteration and mean_square_error >= error_thresold:
            # synaptic weights correction
            for k,data in enumerate(dataset):
                #print(f'iteration {iteration} - data {k}: {[float("%.6f" % weight) for weight in self.weights]}')
                output = 0
                for i,input in enumerate(data.inputs):
                    output += input * self.weights[i]
                local_error = data.currentLabel - output
                for i,weight in enumerate(self.weights):
                    self.weights[i] = weight + self.learning_rate * local_error * data.inputs[i]

            # mean square error computation
            outputs = []
            for data in dataset:
                output = 0
                for i,input in enumerate(data.inputs):
                    output += input * self.weights[i]
                outputs.append(output)
            mean_square_error = 0
            for i,data in enumerate(dataset):
                mean_square_error += (data.currentLabel - outputs[i]) ** 2
            mean_square_error /= 2 * K
            
            iteration += 1

        return iteration-1,mean_square_error
    
    def getLigneDecisionPLotly(self):
        # Calculate the slope and intercept of the decision boundary
        slope = -self.weights[1] / self.weights[2]
        intercept = -self.weights[0] / self.weights[2]
        
        # On a maintenant besoin d'élonger la droite (car elle ne se base que sur deux points)
        # On crée une série de valeurs X
        xSeuillage = np.linspace(-10, 10, 400)
        
        # On calcule pour chaque point son ordonnée
        ySeuillage =  slope * xSeuillage + intercept
        
        # On crée une droite de seuillage
        seuillage = go.Scatter(x=xSeuillage, y=ySeuillage, mode='lines', name='Droite de seuillage')
        
        return seuillage

def openFile(filename):
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, filename)
    
    x1 = pd.read_csv(filename, header=None)
    datas = x1.values.tolist()
    
    return convertToLabeledData(datas)

def convertToLabeledData(data):
    labeledData = []
    for d in data:
        labeledData.append(LabeledData(d[:2], d[2:]))
    return labeledData

def showPlotly(ligneDecision, data):
    x_data = []
    y_data = []
    color_data = []
    
    for d in data:
        x_data.append(d.inputs[1])
        y_data.append(d.inputs[2])
        color_data.append('green' if d.label[0] > 0 else 'red')
    
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

def main():
    processDataset("Datas/table_2_1.csv")
    processDataset("Datas/table_2_9.csv")
    processDataset("Datas/table_2_10.csv")
    
def processDataset(filename):
    dataset = openFile(filename)

    sigma = 0.1
    rate = 0.01
    ada = Adaline(rate, 2, sigma)
    tuple = ada.train(dataset, 1000, 0.01)
    print(ada)
    print(f'nb iteration: {tuple[0]}, mean square error: {tuple[1]:.6f}')
    
    showPlotly(ada.getLigneDecisionPLotly(), dataset)



if __name__ == "__main__":
    main()