import numpy as np
from model_development import KNNClassifier
from metrics import Metrics

class Holdout:
    """
    Modella la tecnica di holdout per la suddivisione di un dataset in training set e test set.
    """

    def __init__(self, data, target, metrics, k, train_size=0.8):
        """
        Inizializza l'oggetto Holdout con i dati, il target, le metriche, il valore di k e la dimensione del training set.

        Parameters:
        data (DataFrame): Il dataset da suddividere.
        target (Series): Le etichette del dataset.
        metrics (list): La lista delle metriche da calcolare.
        k (int): Il numero di vicini da considerare nel classificatore k-NN.
        train_size (float): La proporzione di dati da utilizzare per il training set (default è 0.8).
        """
        self.data = data
        self.target = target
        self.metrics = metrics
        self.k = k
        self.train_size = train_size

    def split(self):
        """
        Suddivide il dataset in training set e test set utilizzando la tecnica di holdout.

        Returns:
        tuple: Contiene il training set, il test set, le etichette del training set e le etichette del test set.
        """
        # Seleziona gli indici per il training set
        train_index = np.random.choice(self.data.index, size=int(self.train_size * len(self.data)), replace=False)
        # Seleziona gli indici per il test set
        test_index = self.data.index[~self.data.index.isin(train_index)]

        # Crea il training set e il test set
        train = self.data.loc[train_index]
        test = self.data.loc[test_index]
        train_target = self.target.loc[train_index]
        test_target = self.target.loc[test_index]

        # Converte i DataFrame e le Series in array numpy
        train = train.to_numpy()
        train_target = train_target.to_numpy()
        test = test.to_numpy()
        test_target = test_target.to_numpy()

        return train, test, train_target, test_target

    def evaluate(self):
        """
        Valuta le performance del modello k-NN utilizzando la tecnica di holdout.

        Returns:
        dict: Le metriche calcolate per il modello.
        """
        # Suddivide i dati in training set e test set
        train, test, train_target, test_target = self.split()

        # Inizializza e addestra il modello k-NN
        knn = KNNClassifier(self.k)
        knn.fit(train, train_target)
        # Predice le etichette per il test set
        predictions = knn.predict(test)

        # Identifica i valori unici delle etichette (label)
        unique_labels = np.unique(self.target)
        if len(unique_labels) != 2:
            raise ValueError("Il modello attualmente supporta solo classificazioni binarie.")
        # Determina le etichette positive e negative
        positive_label = unique_labels[1]
        negative_label = unique_labels[0]

        # Calcola le metriche: true positive, true negative, false positive, false negative
        true_positive = np.sum((predictions == positive_label) & (test_target == positive_label))
        true_negative = np.sum((predictions == negative_label) & (test_target == negative_label))
        false_positive = np.sum((predictions == positive_label) & (test_target == negative_label))
        false_negative = np.sum((predictions == negative_label) & (test_target == positive_label))

        # Crea un oggetto Metrics e calcola le metriche richieste
        metrics = Metrics(true_positive, true_negative, false_positive, false_negative, self.metrics)
        output_metrics = metrics.get_metrics()
        # Salva le metriche calcolate
        metrics.save_metrics(output_metrics)

        return output_metrics  # Restituisce le metriche calcolate
