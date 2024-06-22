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
            il dataset da suddividere
        target : pandas.Series
            la serie di valori target
        metrics : list
            la lista di metriche da calcolare
        k : int
            il valore di k per il modello KNN
        K : int
            il numero di fold
        fold : list
            una lista contenente:
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
        Suddivide il dataset in training set e test set.

        Returns
        -------
         fold : list
            una lista contenente:
            - il training set
            - il test set
            - i valori target del training set
            - i valori target del test set
        """
        indices = np.random.permutation(self.data.index)
        fold_size = int(len(self.data) / self.K)
        for i in range(self.K):
            test_index = indices[i * fold_size:(i + 1) * fold_size]
            train_index = indices[~np.isin(indices, test_index)]

            train = self.data.loc[train_index]
            test = self.data.loc[test_index]

            train_target = self.target.loc[train_index]
            test_target = self.target.loc[test_index]

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
        None
        """
        true_positive_list = []
        true_negative_list = []
        false_positive_list = []
        false_negative_list = []
        self.split()

        for fold in self.fold:
            train, test, train_target, test_target = fold
            knn = KNNClassifier(self.k)
            knn.fit(train, train_target)
            predictions = knn.predict(test)

            unique_labels = np.unique(self.target)
            if len(unique_labels) != 2:
                raise ValueError("Il modello attualmente supporta solo classificazioni binarie.")
            positive_label = unique_labels[1]
            negative_label = unique_labels[0]

            true_positive = np.sum((predictions == positive_label) & (test_target == positive_label))
            true_negative = np.sum((predictions == negative_label) & (test_target == negative_label))
            false_positive = np.sum((predictions == positive_label) & (test_target == negative_label))
            false_negative = np.sum((predictions == negative_label) & (test_target == positive_label))

            true_positive_list.append(true_positive)
            true_negative_list.append(true_negative)
            false_positive_list.append(false_positive)
            false_negative_list.append(false_negative)

        metrics = Metrics(true_positive_list, true_negative_list, false_positive_list, false_negative_list, self.metrics)
        output_metrics = metrics.get_metrics(self.K)
        metrics.save_metrics(output_metrics)
        metrics.metrics_plot(output_metrics)
