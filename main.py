import pandas as pd
import sys
from DataProcessing.data_processing import DataFrameCleaner  # Importa la nuova classe
from DataProcessing.standardizer import Standardizer
from evaluate_holdout import Holdout
from kfold import KFoldCrossValidation
from input_method import Input

def main():
    """
    Funzione principale che gestisce il flusso del programma:
    - Ottiene i parametri di input dall'utente.
    - Esegue il preprocessing dei dati (rimozione duplicati, gestione dei valori mancanti e degli outlier).
    - Standardizza i dati.
    - Esegue la valutazione del modello utilizzando Holdout o K-Fold Cross Validation.
    - Stampa e salva i risultati delle metriche.
    """
    # Ottiene i parametri di input dall'utente
    input_params = Input()
    input_params.get_input()

    data = input_params.data
    target = data[input_params.target_column]
    data = data.drop(columns=[input_params.target_column])

    # Esegue il preprocessing dei dati
    try:
        cleaner = DataFrameCleaner(data)  # Inizializza la classe con il DataFrame
        data = cleaner.remove_duplicates()  # Rimuove i duplicati
        data = cleaner.handle_missing_values()  # Gestisce i valori mancanti
        data = cleaner.handle_outliers()  # Gestisce i valori anomali
    except Exception as e:
        print(f"Errore durante il preprocessing delle features: {e}")
        sys.exit(1)

    # Standardizza i dati
    try:
        standardizer = Standardizer()
        data_standardized, target = standardizer.standardize(pd.concat([data, target], axis=1))
    except Exception as e:
        print(f"Errore durante la standardizzazione delle features: {e}")
        sys.exit(1)

    # Aggiunge la colonna target ai dati standardizzati
    data_standardized['Class'] = target

    # Salva il dataset standardizzato su un file CSV
    standardized_file = 'breast_cancer_standardized.csv'
    data_standardized.to_csv(standardized_file, index=False)
    print(f'Il dataset standardizzato è stato salvato come {standardized_file}')

    # Esegue la valutazione del modello
    try:
        if input_params.evaluation == 1:
            holdout_evaluator = Holdout(data_standardized.drop(columns=['Class']), target, input_params.metrics, input_params.k, input_params.training / 100)
            metrics = holdout_evaluator.evaluate()  # Valutazione del modello con Holdout
        elif input_params.evaluation == 2:
            kfold_evaluator = KFoldCrossValidation(data_standardized.drop(columns=['Class']), target, input_params.metrics, input_params.k, input_params.K)
            metrics = kfold_evaluator.evaluate()  # Valutazione del modello con K-Fold Cross Validation
        else:
            print("Metodo di valutazione non valido. Scegliere 1 per Holdout o 2 per KFold Cross Validation.")
            sys.exit(1)
    except Exception as e:
        print(f"Errore durante la valutazione del modello: {e}")
        sys.exit(1)

    # Stampa le metriche calcolate
    print("Metrics Calculated:", metrics)

# Verifica se lo script viene eseguito direttamente
if __name__ == "__main__":
    main()
