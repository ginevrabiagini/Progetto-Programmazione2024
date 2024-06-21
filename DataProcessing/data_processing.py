import pandas as pd

class DataFrameCleaner:

    def __init__(self, df):
        self.df = df

    # Funzione per eliminare i duplicati
    # Riceve un DataFrame e restituisce un DataFrame senza duplicati
    # Stampa il numero di duplicati rimossi
    def remove_duplicates(self):
        initial_shape = self.df.shape
        self.df = self.df.drop_duplicates()
        final_shape = self.df.shape
        if initial_shape[0] == final_shape[0]:
            print('Non ci sono duplicati')
        else:
            print(f"Rimossi {initial_shape[0] - final_shape[0]} duplicati")
        return self.df

    # Funzione per gestire i valori mancanti
    # Riceve un DataFrame e restituisce un DataFrame con i valori mancanti sostituiti dalla mediana della colonna
    # Stampa i valori mancanti iniziali e un messaggio dopo la sostituzione
    def handle_missing_values(self):
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            print("Valori mancanti:\n", missing_values)
            for column in self.df.columns:
                if self.df[column].isnull().sum() > 0:
                    median_value = self.df[column].median()
                    self.df[column] = self.df[column].fillna(median_value)
            print("Valori mancanti sostituiti con la mediana della colonna.")
        else:
            print('Non ci sono valori mancanti')
        return self.df

    # Funzione per gestire i valori anomali
    # Riceve un DataFrame e restituisce un DataFrame con i valori anomali sostituiti dalla mediana della colonna
    # Stampa i valori anomali iniziali e un messaggio dopo la sostituzione
    def handle_outliers(self):
        out_of_range = self.df.iloc[:, 1:-1].apply(lambda x: (x < 1) | (x > 10)).sum()  # iloc seleziona le colonne e lambda controlla se i valori mancanti sono <1 o >10
        if out_of_range.any():
            print("Valori anomali:\n", out_of_range)
            for column in self.df.columns[1:-1]:  # Escludiamo la prima e l'ultima colonna
                self.df.loc[(self.df[column] < 1) | (self.df[column] > 10), column] = self.df[column].median()
            print("Valori anomali sostituiti con la mediana della colonna.")
        else:
            print('Non ci sono valori anomali')
        return self.df
