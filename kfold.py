import numpy as np
from model_development import KNNClassifier
from metrics import Metrics

class KFoldCrossValidation:
    """
    Modella la tecnica di k-fold cross validation per la suddivisione di un dataset in training set e test set
    """

    def __init__(self, data, target, metrics, k, K, fold=[]):
        """
        Inizializza la classe KFoldCrossValidation con i dati, il target, le metriche, il numero di vicini per KNN,
        il numero di fold e la lista dei fold

        Parameters:
        data (DataFrame): Il DataFrame contenente i dati del dataset
        target (Series): Il Series contenente le etichette target del dataset
        metrics (list): La lista di metriche da calcolare
        k (int): Il numero di vicini da considerare per il KNN
        K (int): Il numero di fold
        fold (list): Una lista contenente i fold:
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
        Suddivide il dataset in K fold, ognuno contenente un training set e un test set

        Returns:
        list: Una lista contenente i fold:
            - il training set
            - il test set
            - i valori target del training set
            - i valori target del test set
        """
        # Permuta gli indici del dataset
        indices = np.random.permutation(self.data.index)

        # Generazione dei fold
        fold_size = int(len(self.data) / self.K)
        for i in range(self.K):
            # Seleziona gli indici per il test set e per il training set
            test_index = indices[i * fold_size:(i + 1) * fold_size]
            train_index = indices[~np.isin(indices, test_index)]

            # Generazione dei train set e test set
            train = self.data.loc[train_index]
            test = self.data.loc[test_index]

            # Generazione dei target
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
        Valuta le performance del modello KNN con k-fold cross validation

        Returns:
        dict: Le metriche aggregate calcolate dalla valutazione del modello

        Stampa e salva le metriche aggregate, e visualizza un grafico delle metriche
        """
        true_positive_list = []
        true_negative_list = []
        false_positive_list = []
        false_negative_list = []
        self.split()

        for i in range(self.K):
            # Ottiene i set di addestramento e test per il fold corrente
            train, test, train_target, test_target = self.fold[i]

            # Creazione di un oggetto KNNClassifier con i parametri specificati
            knn = KNNClassifier(self.k)

            # Addestramento del modello KNN sul set di addestramento corrente
            knn.fit(train, train_target)

            # Predizione dei valori target sul set di test corrente
            predictions = knn.predict(test)

            # Calcolo delle metriche: true positive, true negative, false positive, false negative
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

            # Aggiunge i risultati del fold alle rispettive liste
            true_positive_list.append(true_positive)
            true_negative_list.append(true_negative)
            false_positive_list.append(false_positive)
            false_negative_list.append(false_negative)

        # Calcola e salva le metriche aggregate e visualizza un grafico
        metrics = Metrics(true_positive_list, true_negative_list, false_positive_list, false_negative_list, self.metrics)
        output_metrics = metrics.get_metrics(self.K)
        metrics.save_metrics(output_metrics)
        metrics.metrics_plot(output_metrics)

        return output_metrics