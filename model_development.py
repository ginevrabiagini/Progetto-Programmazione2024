import numpy as np
from collections import Counter
import random
import pandas as pd
from sklearn.model_selection import train_test_split

class KNNClassifier:
    def __init__(self, k=3):
        """
        Inizializzazione del classificatore k-NN con un determinato valore di k.

        Parameters:
        k (int): Numero di vicini più prossimi da considerare.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Addestramento del classificatore memorizzando i dati di training.

        Parameters:
        X_train (numpy.ndarray): Matrice delle caratteristiche di training.
        y_train (numpy.ndarray): Array delle etichette di training.
        """
        # Memorizza la matrice delle caratteristiche di training
        self.X_train = X_train
        # Memorizza l'array delle etichette di training
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predizione delle etichette per i dati di test.

        Parameters:
        X_test (numpy.ndarray): Matrice delle caratteristiche di test.

        Returns:
        numpy.ndarray: Array delle etichette predette per i dati di test.
        """
        # Calcola le predizioni per ciascun campione di test utilizzando la funzione _predict
        predictions = [self._predict(x) for x in X_test]
        # Converte la lista delle predizioni in un array numpy e lo restituisce
        return np.array(predictions)

    def _predict(self, x):
        """
        Predizione dell'etichetta per un singolo campione di test.

        Parameters:
        x (numpy.ndarray): Singolo campione di test.

        Returns:
        int: Etichetta predetta per il campione di test.
        """
        # Calcola la distanza euclidea tra il campione di test x e tutti i campioni di training
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        # Ottiene gli indici dei k campioni di training più vicini
        k_indices = np.argsort(distances)[:self.k]
        # Ottiene le etichette dei k campioni di training più vicini
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Trova l'etichetta più comune tra i vicini più prossimi
        most_common = Counter(k_nearest_labels).most_common()
        # Trova la frequenza massima delle etichette più comuni
        max_count = most_common[0][1]
        # Trova tutte le etichette che hanno la frequenza massima
        candidates = [label for label, count in most_common if count == max_count]
        # Se c'è più di un candidato, ne sceglie uno a caso; altrimenti, restituisce l'unico candidato
        return random.choice(candidates) if len(candidates) > 1 else candidates[0]

if __name__ == "__main__":
    # Caricamento del dataset
    dataset = pd.read_csv("breast_cancer_standardized.csv")

    # Divisione del dataset in caratteristiche (X) e target (y)
    X = dataset.drop(columns=['Class']).values
    y = dataset['Class'].values

    # Divisione del dataset in training set e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inizializzazione del classificatore k-NN con k=3
    knn = KNNClassifier(k=3)
    # Addestramento del classificatore
    knn.fit(X_train, y_train)

    # Calcolo delle predizioni
    predictions = knn.predict(X_test)

    # Stampa delle predizioni
    print("Predizioni per il set di test:")
    print(predictions)
