import numpy as np
from model_development import KNNClassifier
from metrics import Metrics

class Holdout:
    """
    Modella la tecnica di holdout per la suddivisione di un dataset in training set e test set.
    """

    def __init__(self, data, target, metrics, k, train_size=0.8):
        self.data = data
        self.target = target
        self.metrics = metrics
        self.k = k
        self.train_size = train_size

    def split(self):
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
        train, test, train_target, test_target = self.split()

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

        metrics = Metrics(true_positive, true_negative, false_positive, false_negative, self.metrics)
        output_metrics = metrics.get_metrics()
        metrics.save_metrics(output_metrics)

        return output_metrics  # Assicurati di restituire le metriche calcolate
