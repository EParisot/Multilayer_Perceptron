import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata


DATA_PATH = 'data'

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """Dérivée de la fonction sigmoid."""
    return sigmoid(x) * (1.0 - sigmoid(x))


def to_one_hot(y, k):
    """Convertit un entier en vecteur "one-hot".

    to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0)

    """
    one_hot = np.zeros(k)
    one_hot[y] = 1
    return one_hot



class Layer:
    """Une seule couche de neurones."""
    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size

        # Les poids sont représentés par une matrice de n lignes
        # et m colonnes. n = le nombre de neurones, m = le nombre de
        # neurones dans la couche précédente.
        self.weights = np.random.randn(size, input_size)

        # Un biais par neurone
        self.biases = np.random.randn(size)

    # Résultat du calcul de chaque neurone.
    # Il est important de noter que `data` est un vecteur (normalement, de
    # longueur `self.input_size`, et que nous retournons un vecteur de
    # taille `self.size`.
    def forward(self, data):
        aggregation = self.aggregation(data)
        activation = self.activation(aggregation)
        return activation

    # Calcule la somme des entrées pondérées + biais pour chaque neurone.
    # Plutôt que d'utiliser une boucle for, nous tirons parti du calcul
    # matriciel qui permet d'effectuer toutes ces opérations d'un coup.
    def aggregation(self, data):
        return np.dot(self.weights, data) + self.biases

    # Passe les valeurs aggrégées dans la moulinette de la fonction
    # d'activation.
    # `x` est un vecteur de longueur `self.size`, et nous retournons un
    # vecteur de même dimension.
    def activation(self, x):
        return sigmoid(x)

    # Dérivée de la fonction d'activation.
    def activation_prime(self, x):
        return sigmoid_prime(x)

    # Mise à jour des poids à partir du gradient (algo du gradient)
    def update_weights(self, gradient, learning_rate):
        self.weights -= learning_rate * gradient

    # Idem mais avec les biais
    def update_biases(self, gradient, learning_rate):
        self.biases -= learning_rate * gradient


class Network:
    """Un réseau constitué de couches de neurones."""
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layers = []

    def add_layer(self, size):
        if len(self.layers) > 0:
            input_dim = self.layers[-1].size
        else:
            input_dim = self.input_dim

        self.layers.append(Layer(size, input_dim))

    # Propage les données d'entrée d'une couche à l'autre.
    def feedforward(self, input_data):
        activation = input_data
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    # Retourne l'index du neurone de sortie qui a la plus haute valeur, ce
    # qui revient à indiquer quelle classe est sélectionnée par le réseau.
    def predict(self, input_data):
        return np.argmax(self.feedforward(input_data))

    # Évalue la performance du réseau à partir d'un set d'exemples.
    # Retourne un nombre entre 0 et 1.
    def evaluate(self, X, Y):
        results = [1 if self.predict(x).all() == y.all() else 0 for (x, y) in zip(X, Y)]
        accuracy = sum(results) / len(results)
        return accuracy

    # Fonction d'entraînement du modèle.
    # Comme décrit dans le billet, nous allons faire tourner la
    # rétropropagation sur un certain nombre d'exemples (batch_size) avant
    # de calculer un gradient moyen, et de mettre à jour les poids.
    def train(self, X, Y, steps=30, learning_rate=0.3, batch_size=10):
        n = Y.size
        for i in range(steps):
            # Mélangeons les données parce que… parce que.
            X, Y = shuffle(X, Y)
            for batch_start in range(0, n, batch_size):
                X_batch, Y_batch = X[batch_start:batch_start + batch_size], Y[batch_start:batch_start + batch_size]
                self.train_batch(X_batch, Y_batch, learning_rate)

    # Cette fonction combine les algos du retropropagation du gradient +
    # gradient descendant.
    def train_batch(self, X, Y, learning_rate):
        # Initialise les gradients pour les poids et les biais.
        weight_gradient = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]

        # On fait tourner l'algo de rétropropagation pour calculer les
        # gradients un certain nombre de fois. On fera la moyenne ensuite.
        for (x, y) in zip(X, Y):
            new_weight_gradient, new_bias_gradient = self.backprop(x, y)
            weight_gradient = [wg + nwg for wg, nwg in zip(weight_gradient, new_weight_gradient)]
            bias_gradient = [bg + nbg for bg, nbg in zip(bias_gradient, new_bias_gradient)]

        # C'est ici qu'on calcule les moyennes des gradients calculés
        avg_weight_gradient = [wg / Y.size for wg in weight_gradient]
        avg_bias_gradient = [bg / Y.size for bg in bias_gradient]

        # Il ne reste plus qu'à mettre à jour les poids et biais en
        # utilisant l'algo du gradient descendant.
        for layer, weight_gradient, bias_gradient in zip(self.layers,
                                                         avg_weight_gradient,
                                                         avg_bias_gradient):
            layer.update_weights(weight_gradient, learning_rate)
            layer.update_biases(bias_gradient, learning_rate)

    # L'algorithme de rétropropagation du gradient.
    # C'est là que tout le boulot se fait.
    def backprop(self, x, y):

        # On va effectuer une passe vers l'avant, une passe vers l'arrière
        # On profite de la passe vers l'avant pour stocker les calculs
        # intermédiaires, qui seront réutilisés par la suite.
        aggregations = []
        activation = x
        activations = [activation]

        # Propagation pour obtenir la sortie
        for layer in self.layers:
            aggregation = layer.aggregation(activation)
            aggregations.append(aggregation)
            activation = layer.activation(aggregation)
            activations.append(activation)

        # Calculons la valeur delta (δ) pour la dernière couche
        # en appliquant les équations détaillées plus haut.
        target = y
        delta = self.get_output_delta(aggregation, activation, target)
        deltas = [delta]

        # Phase de rétropropagation pour calculer les deltas de chaque
        # couche
        # On utilise une implémentation vectorielle des équations.
        nb_layers = len(self.layers)
        for l in reversed(range(nb_layers - 1)):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            activation_prime = layer.activation_prime(aggregations[l])
            delta = activation_prime * np.dot(next_layer.weights.transpose(), delta)
            deltas.append(delta)

        # Nous sommes parti de l'avant-dernière couche pour remonter vers
        # la première. deltas[0] contient le delta de la dernière couche.
        # Nous l'inversons pour faciliter la gestion des indices plus tard.
        deltas = list(reversed(deltas))

        # On utilise maintenant les deltas pour calculer les gradients.
        weight_gradient = []
        bias_gradient = []
        for l in range(len(self.layers)):

            # Notez que l'indice des activations est « décalé », puisque
            # activation[0] contient l'entrée (x), et pas l'activation de
            # la première couche.
            prev_activation = activations[l]
            weight_gradient.append(np.outer(deltas[l], prev_activation))
            bias_gradient.append(deltas[l])

        return weight_gradient, bias_gradient

    # Calcule le delta pour la dernière couche, en utilisant
    # les dernières valeurs d'aggregation, d'activation, et la valeur
    # cible.
    # Notez que lorsque l'on utilise l'entropie croisée pour fonction de
    # coût, l'équation de calcul de delta peut-être simplifiée pour aboutir
    # au résultat ci dessous.
    # Cf http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function
    def get_output_delta(self, z, a, target):
        return a - target






def read_data(data_file, sep, labels_col_idx):
    with open(data_file, mode="r") as f:
        for j, line in enumerate(f):
            if j == 0:
                data = [[] for elem in line.split(sep)]
            for i, feat in enumerate(line.split(sep)):
                data[i].append(feat[:-1] if feat.endswith("\n") else feat)
    X = data[1:labels_col_idx] + data[labels_col_idx+1:]
    X = normalise(X).T
    Y = data[labels_col_idx]
    Y, nb_class = str_to_int(Y)
    Y = to_categorical(Y, nb_class)
    return X, Y

def to_categorical(Y, nb_class):
    values = range(nb_class)
    cat_Y = []
    for i, elem in enumerate(Y):
        if elem in values:
            cat_Y.append(np.zeros(nb_class))
            cat_Y[i][values.index(elem)] = 1
    return np.array(cat_Y)

def normalise(X):
    norm_X = []
    for x in X:
        x = np.array(x).astype(np.float)
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        norm_X.append(x)
    return np.array(norm_X)

def str_to_int(Y):
    uniques = []
    for elem in Y:
        if elem not in uniques:
            uniques.append(elem)
    for i, elem in enumerate(Y):
        Y[i] = uniques.index(elem)
    return np.array(Y), len(uniques)

if __name__ == '__main__':

    X, Y = read_data("data.csv", ",", 1)

    print(X.shape, Y.shape)

    net = Network(input_dim=len(X[0]))
    net.add_layer(8)
    net.add_layer(4)
    net.add_layer(2)

    accuracy = net.evaluate(X, Y)
    print('Performance initiale : {:.2f}%'.format(accuracy * 100.0))

    for i in range(30):
        net.train(X, Y, steps=1, learning_rate=3.0)
        accuracy = net.evaluate(X, Y)
        print('Nouvelle performance : {:.2f}%'.format(accuracy * 100.0))