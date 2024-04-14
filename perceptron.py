from typing import List
import plotly.graph_objects as go
import pandas as pd

class modele:
    # Liste de poids a taille variable
    listPoids:List[float] = []
    # Nombre d'entrées
    nbrEntrees:int = 0
    
    tauxApprentissage:float = 1
    maxIterations:int = 50
    
    debug:bool = False
    
    def __init__(self, nbrEntrees, tauxApprentissage=0.001, nbrIterationMax=1000, debug = False):
        self.nbrEntrees = nbrEntrees
        self.tauxApprentissage = tauxApprentissage
        self.maxIterations = nbrIterationMax
        self.debug = debug
        
        # On initialise les poids à nbrEntrees + 1 (ne pas oublier w0)
        for i in range(nbrEntrees+1):
            self.listPoids.append(0)
    
    def processData(self, datas:List[bool]):
        # On commence par le poid w0
        p = self.listPoids[0] * 1
        
        for i in range(self.nbrEntrees):
            p += self.listPoids[i+1] * datas[i] # i+1 car le poids 0 est w0
        return p >= 0


    def train(self, datas:List[List[bool]]):
        nbrColonnesDataset = len(datas[0]) - 1 # Car on ne prend pas en compte la dernière colonne (valeur attendue)
        
        # On vérifie qu'on a le bon nombre d'entrées
        if nbrColonnesDataset != self.nbrEntrees:
            print("Mauvaise taille de données d'entrée")
            return -1
        
        # Initialisation des variables
        iteration = 0
        endIterations = False
        
        # Début du traitement
        while endIterations == False:
            # A chaque itération, le nbr d'erreur est mis à 0
            error = 0
            iteration += 1

            # On passe par chaque ligne du jeu de donnée
            for data in datas:                
                # Une donnée est traitée par le modèle 
                result = self.processData(data)

                if self.debug:
                    self.debugTrain(data, iteration, result, error)
                    
                # Si on a une erreur, on update les poids et incrémente le compteur d'erreur
                d = data[len(data)-1]
                if result != d:
                    error += 1
                    self.processError(result, data, d)
            
            # En fin de boucle, on vérifie s'il y a eu des erreurs (et si on a dépassé le nbr max d'itérations)
            if error == 0 or iteration >= self.maxIterations:
                endIterations = True
        
        return iteration
    
    def debugTrain(self, data:List[bool], iteration:int, result:bool, error:int):
        affichage = "Itération " + str(iteration) + "\n"
        affichage += "Données : "
        
        for d in data[:-1]:
            affichage += str(d) + " "
                    
        affichage += " | Attendu : {} | Résultat : {} | Errors : {}".format(data[len(data)-1], int(result), error)
        
        print(affichage)
                
        

    def processError(self, result:float, data:List[bool], d:bool):
        error = d - result

        self.listPoids[0] += self.tauxApprentissage * error * 1        
        for index in range(self.nbrEntrees):
            self.listPoids[index + 1] += self.tauxApprentissage * error * data[index]
        

def showPlotly(x_line, y_line, data):
    x_data = []
    y_data = []
    
    for d in data:
        x_data.append(d[0])
        y_data.append(d[1])
    
    # Créer un scatter plot pour les points
    scatter_trace = go.Scatter(x=x_data, y=y_data, mode='markers', name='Values')

    # Créer une ligne indépendante des points
    line_trace = go.Scatter(x=x_line, y=y_line, mode='lines', name='Droite de seuillage')

    # Créer la figure
    fig = go.Figure()

    # Ajouter les traces à la figure
    fig.add_trace(scatter_trace)
    fig.add_trace(line_trace)

    # Ajouter une mise en page
    fig.update_layout(title='Scatter Plot with Independent Line',
                    xaxis=dict(title='X-Axis'),
                    yaxis=dict(title='Y-Axis'))

    # Afficher la figure
    fig.show()

def openFile(filename):
    x1 = pd.read_csv(filename, header=None)
    return x1.values.tolist()

def affichageFinal(model, local_datas, iteration):
    print("Il a fallu {} itérations pour résoudre le modèle".format(iteration))
    print("""Les poids sont : 
            - w0 : {}
            - w1 : {}
            - w2 : {}
          """.format(model.listPoids[0], model.listPoids[1], model.listPoids[2]))
    
    showPlotly([0, - model.listPoids[0] / model.listPoids[2]], [-model.listPoids[0]/ model.listPoids[1], 0], local_datas)

if __name__ == "__main__":
    # Table 2.1 : porte ET
    datas = [[0,0,0], [0,1,0], [1,0,0], [1,1,1]]

    print("Entrainement du modèle 1 : porte ET")
    
    m = modele(2)
    it = m.train(datas)
    affichageFinal(m, datas, it)    
    