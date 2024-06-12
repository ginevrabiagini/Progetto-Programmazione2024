import pandas as pd
import sys
from DataProcessing.data_processing import remove_duplicates, handle_missing_values, handle_outliers
from DataProcessing.standardizer import Standardizer

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_csv>")
        sys.exit(1)

    # Caricare il file CSV
    dataset = sys.argv[1]
    data = pd.read_csv(dataset)

    # Verifica e rimozione duplicati
    data = remove_duplicates(data)

    # Gestione valori mancanti
    data = handle_missing_values(data)

    # Gestione valori anomali
    data = handle_outliers(data)

    # Salvataggio del dataset pulito
    cleaned_file = 'breast_cancer_cleaned.csv'
    data.to_csv(cleaned_file, index=False)
    print(f'Il dataset aggiornato è stato salvato come {cleaned_file}')

    # Caricamento del dataset pulito
    data = pd.read_csv(cleaned_file)

    # Standardizzazione del dataset
    standardizer = Standardizer()
    data_standardized, target = standardizer.standardize(data)

    # Stampa del risultato finale
    print("Features standardizzate:")
    print(data_standardized.head())
    print("\nTarget:")
    print(target.head())

    # Salvataggio del dataset standardizzato su un file CSV
    standardized_file = 'breast_cancer_standardized.csv'
    data_standardized['Class'] = target
    data_standardized.to_csv(standardized_file, index=False)
    print(f'Il dataset standardizzato è stato salvato come {standardized_file}')

if __name__ == "__main__":
    main()
