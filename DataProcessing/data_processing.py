import pandas as pd

class DataFrameCleaner:
    """
    Classe per il preprocessing dei dati, inclusa la rimozione dei duplicati,
    la gestione dei valori mancanti e la gestione dei valori anomali.
    """


    def remove_duplicates(df):
        """
        Rimuove i duplicati dal DataFrame.

        Parameters:
        df (DataFrame): Il DataFrame contenente i dati.

        Returns:
        DataFrame: Il DataFrame senza duplicati.
        """
        initial_shape = df.shape
        df = df.drop_duplicates()
        final_shape = df.shape
        if initial_shape[0] == final_shape[0]:
            print('Non ci sono duplicati')
        else:
            print(f"Rimossi {initial_shape[0] - final_shape[0]} duplicati")
        return df


    def handle_missing_values(df):
        """
        Gestisce i valori mancanti nel DataFrame sostituendoli con la mediana della colonna.

        Parameters:
        df (DataFrame): Il DataFrame contenente i dati.

        Returns:
        DataFrame: Il DataFrame con i valori mancanti sostituiti.
        """
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("Valori mancanti:\n", missing_values)
            for column in df.columns:
                if df[column].isnull().sum() > 0:
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
            print("Valori mancanti sostituiti con la mediana della colonna.")
        else:
            print('Non ci sono valori mancanti')
        return df


    def handle_outliers(df):
        """
        Gestisce i valori anomali nel DataFrame sostituendoli con la mediana della colonna.

        Parameters:
        df (DataFrame): Il DataFrame contenente i dati.

        Returns:
        DataFrame: Il DataFrame con i valori anomali sostituiti.
        """
        df = df.copy()  # Evita il SettingWithCopyWarning
        out_of_range = df.iloc[:, 1:-1].apply(lambda x: (x < 1) | (x > 10)).sum()
        if out_of_range.any():
            print("Valori anomali:\n", out_of_range)
            for column in df.columns[1:-1]:
                df.loc[(df[column] < 1) | (df[column] > 10), column] = df[column].median()
            print("Valori anomali sostituiti con la mediana della colonna.")
        else:
            print('Non ci sono valori anomali')
        return df
