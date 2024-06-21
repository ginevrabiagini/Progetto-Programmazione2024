import pandas as pd
import numpy as np

class Standardizer:
    """
    Classe per la standardizzazione dei valori delle features presenti nel dataset
    """

    def standardize(self, dataset):
        """
        Standardizza le features del dataset portando la media a 0 e la deviazione standard a 1

        Parameters:
        dataset (DataFrame): Il DataFrame contenente il dataset

        Returns:
        DataFrame, Series: Il DataFrame con le features standardizzate e la target label
        """
        # Separa la colonna 'Class' dal resto del dataset
        target = dataset['Class']
        features = dataset.drop(columns=['Class'])

        # Standardizzazione delle colonne delle features
        for col in features.columns:
            if np.issubdtype(features[col].dtype, np.number):
                features[col] = (features[col] - features[col].mean()) / features[col].std()

        return features, target

    def split(self, dataset):
        """
        Divide il dataset in features e target label

        Parameters:
        dataset (DataFrame): Il DataFrame contenente il dataset

        Returns:
        DataFrame, Series: Il DataFrame con le features e il Series con la target label
        """
        # Seleziona tutte le colonne tranne 'Class' come features
        data = dataset.drop(columns=['Class'])
        # Seleziona solo la colonna 'Class' come target label
        target = dataset['Class']

        return data, target
""""
# Caricamento del dataset
data = pd.read_csv('breast_cancer_cleaned.csv')

# Creazione di un'istanza di Standardizer e standardizzazione del dataset
standardizer = Standardizer()
data_standardized, target = standardizer.standardize(data)

# Stampa del risultato finale
print("Features standardizzate:")
print(data_standardized)
print("\nTarget:")
print(target)

# Salvataggio del dataset standardizzato su un file CSV
data_standardized['Class'] = target
data_standardized.to_csv('breast_cancer_standardized.csv', index=False)"" """""