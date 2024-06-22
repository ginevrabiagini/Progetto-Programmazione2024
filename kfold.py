import numpy as np
from model_development import KNNClassifier
from metrics import Metrics


class KFoldCrossValidation:
    """
    Modella la tecnica di k-fold cross validation per la suddivisione di un dataset in training set e test set.
    """

    def __init__(self, data, target, metrics, k, K, fold=[]):
        """
        Costruttore

        Parameters
        ----------
        data : pandas.DataFrame
            Il dataset da suddividere
        target : pandas.Series
            La serie di valori target
        metrics : list
            La lista di metriche da calcolare
        k : int
            Il valore di k per il modello KNN
        K : int
            Il numero di fold
        fold : list
            Una lista contenente:
            - il training set
            - il test set
            - i valori target del training set
            - i valori target del test set
        """
        self.data = data
        self.target = target
        self.metrics = metrics
        self.k = k
        self.K = K
        self.fold = fold

    def split(self):
        """
        Suddivide il dataset in K fold per la validazione incrociata.

        Returns
        -------
        fold : list
            Una lista contenente:
            - il training set
            - il test set
            - i valori target del training set
            - i valori target del test set
        """
        # Permuta gli indici del dataset
        indices = np.random.permutation(self.data.index)
        fold_size = int(len(self.data) / self.K)

        # Genera i fold
        for i in range(self.K):
            # Seleziona gli indici per il test set e per il training set
            test_index = indices[i * fold_size:(i + 1) * fold_size]
            train_index = indices[~np.isin(indices, test_index)]

            # Crea il training set e il test set
            train = self.data.loc[train_index]
            test = self.data.loc[test_index]
            train_target = self.target.loc[train_index]
            test_target = self.target.loc[test_index]

            # Converte i DataFrame e le Series in array numpy
            train = train.to_numpy()
            test = test.to_numpy()
            train_target = train_target.to_numpy()
            test_target = test_target.to_numpy()

            self.fold.append([train, test, train_target, test_target])
        return self.fold

    def evaluate(self):
        """
        Valuta le performance del modello KNN con k-fold cross validation.

        Returns
        -------
        dict
            Le metriche calcolate per il modello.
        """
        true_positive_list = []
        true_negative_list = []
        false_positive_list = []
        false_negative_list = []

        # Suddivide il dataset in K fold
        self.split()

        for fold in self.fold:
            train, test, train_target, test_target = fold

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

            true_positive_list.append(true_positive)
            true_negative_list.append(true_negative)
            false_positive_list.append(false_positive)
            false_negative_list.append(false_negative)

        # Crea un oggetto Metrics e calcola le metriche richieste
        metrics = Metrics(true_positive_list, true_negative_list, false_positive_list, false_negative_list,
                          self.metrics)
        output_metrics = metrics.get_metrics(self.K)
        # Salva le metriche calcolate
        metrics.save_metrics(output_metrics)
        metrics.metrics_plot(output_metrics)
