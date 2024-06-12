import pandas as pd

# Caricare il file CSV
dataset = 'breast_cancer.csv'
data = pd.read_csv(dataset)

# Funzione per eliminare i duplicati
def remove_duplicates(df):
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape
    if initial_shape[0] == final_shape[0]:
        print('Non ci sono duplicati')
    else:
        print(f"Rimossi {initial_shape[0] - final_shape[0]} duplicati")
    return df

# Funzione per gestire i valori mancanti
def handle_missing_values(df):
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Valori mancanti:\n", missing_values)
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
        print("Valori mancanti sostituiti con la mediana della colonna.")
    else:
        print('Non ci sono valori mancanti')
    return df

# Funzione per gestire i valori anomali

def handle_outliers(df):
    out_of_range = df.iloc[:, 1:-1].apply(lambda x: (x < 1) | (x > 10)).sum()
    if out_of_range.any():
        print("Valori anomali:\n", out_of_range)
        for column in df.columns[1:-1]:  # Escludiamo la prima e l'ultima colonna
            df.loc[(df[column] < 1) | (df[column] > 10), column] = df[column].median()
        print("Valori anomali sostituiti con la mediana della colonna.")
    else:
        print('Non ci sono valori anomali')
    return df

# Verifica e rimozione duplicati
data = remove_duplicates(data)

# Gestione valori mancanti
data = handle_missing_values(data)

# Gestione valori anomali
data = handle_outliers(data)

# Salvataggio del dataset pulito
data.to_csv('breast_cancer_cleaned.csv', index=False)
print('Il dataset aggiornato è stato salvato come breast_cancer_cleaned.csv')

# Visualizzazione del dataset salvato
cleaned_data = pd.read_csv('breast_cancer_cleaned.csv')
print(cleaned_data.head())