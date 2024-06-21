import numpy as np
from model_development import KNNClassifier
from metrics import Metrics

class Holdout:
    """
    Modella la tecnica di holdout per la suddivisione di un dataset in training set e test set
    """

    def __init__(self, data, target, metrics, k, train_size=0.8):
        """
        Inizializza la classe Holdout con i dati, il target, le metriche, il numero di vicini per KNN e la dimensione del training set

        Parameters:
        data (DataFrame): Il DataFrame contenente i dati del dataset
        target (Series): Il Series contenente le etichette target del dataset
        metrics (str): Le metriche da utilizzare per la valutazione
        k (int): Il numero di vicini da considerare per il KNN
        train_size (float): La proporzione del dataset da utilizzare come training set
        """
        self.data = data
        self.target = target
        self.metrics = metrics
        self.k = k
        self.train_size = train_size

    def split(self):
        """
        Divide il dataset in training set e test set secondo la proporzione specificata

        Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray: I training set, test set, training target e test target
        """
        train_index = np.random.choice(self.data.index, size=int(self.train_size * len(self.data)), replace=False)
        test_index = self.data.index[~self.data.index.isin(train_index)]

        train = self.data.loc[train_index]
        test = self.data.loc[test_index]
        train_target = self.target.loc[train_index]
        test_target = self.target.loc[test_index]

        train = train.to_numpy()
        train_target = train_target.to_numpy()
        test = test.to_numpy()
        test_target = test_target.to_numpy()

        return train, test, train_target, test_target

    def evaluate(self):
        """
        Valuta il modello KNN utilizzando la suddivisione holdout del dataset

        Returns:
        dict: Le metriche calcolate della valutazione del modello

        Stampa le metriche calcolate e salva le metriche su un file
        """
        train, test, train_target, test_target = self.split()

        knn = KNNClassifier(self.k)
        knn.fit(train, train_target)
        predictions = knn.predict(test)

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for i in range(len(predictions)):
            if predictions[i] == 4 and test_target[i] == 4:
                true_positive += 1
            elif predictions[i] == 4 and test_target[i] == 2:
                false_positive += 1
            elif predictions[i] == 2 and test_target[i] == 2:
                true_negative += 1
            elif predictions[i] == 2 and test_target[i] == 4:
                false_negative += 1

        metrics = Metrics(true_positive, true_negative, false_positive, false_negative, self.metrics)
        output_metrics = metrics.get_metrics()
        metrics.save_metrics(output_metrics)

        return output_metrics  # Restituisce le metriche calcolate