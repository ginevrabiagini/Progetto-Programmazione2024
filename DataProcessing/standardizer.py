import pandas as pd
import numpy as np

class Standardizer:
    """
    Classe per la standardizzazione dei valori delle features presenti nel dataset.
    """

    def standardize(self, dataset):
        """
        Standardizza le features del dataset portando la media a 0 e la deviazione standard a 1,
        senza separare la colonna delle label.

        Parameters:
        dataset (DataFrame): Il DataFrame contenente il dataset.

        Returns:
        DataFrame: Il DataFrame con le features standardizzate e le etichette originali.
        """
        label_column = 'Class'
        # Standardizzazione delle colonne delle features, eccetto la colonna delle label
        for col in dataset.columns:
            if col != label_column and np.issubdtype(dataset[col].dtype, np.number):
                dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()

        print("Features standardizzate con successo.")
        return dataset