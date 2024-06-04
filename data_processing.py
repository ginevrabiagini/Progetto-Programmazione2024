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

# Verifica e rimozione duplicati
data = remove_duplicates(data)


# Controllare la presenza di valori mancanti
missing_values = data.isnull().sum()

# Controllare se ci sono valori fuori dal range 1-10 per le caratteristiche (escludendo la colonna "Sample code number" e "Class")
out_of_range = data.iloc[:, 1:-1].apply(lambda x: (x < 1) | (x > 10)).sum()

# Verifica se ci sono valori mancanti
if missing_values.any():
    print("Valori mancanti:\n", missing_values)
else:
    print('Non ci sono valori mancanti')

# Verifica se ci sono valori anomali
if out_of_range.any():
    print("Valori anomali:\n", out_of_range)
else:
    print('Non ci sono valori anomali')

print('il dataset aggiornato è : ',data)

