# Progetto-Programmazione2024
Questo progetto mira a sviluppare un programma capace di determinare la natura benigna o maligna di un tumore, analizzando un dataset contenente informazioni su vari tipi di cellule tumorali. Il programma utilizza un classificatore kNN (k-Nearest Neighbors) per eseguire la classificazione dei dati e valutare l'efficacia del modello.

# Dataset
In questo progetto utilizziamo il dataset breast_cancer.csv il quale fornisce dettagli su alcuni tipi di cellule tumorali. In particolare, il dataset è strutturato come segue:

Numero di Campioni: 683 campioni.

Numero di Caratteristiche: 11 caratteristiche per campione.

Nomi delle Caratteristiche:

Sample code number: Identificativo unico per ogni campione.
Clump Thickness: Spessore del grumo di cellule.
Uniformity of Cell Size: Uniformità delle dimensioni cellulari.
Uniformity of Cell Shape: Uniformità delle forme cellulari.
Marginal Adhesion: Adesione marginale delle cellule.
Single Epithelial Cell Size: Dimensione della singola cellula epiteliale.
Bare Nuclei: Nuclei scoperti.
Bland Chromatin: Cromatina blanda.
Normal Nucleoli: Nucleoli normali.
Mitoses: Tasso di mitosi.
Class: Classificazione del tumore (2 per benigno, 4 per maligno).

# Esecuzione del codice 
Per eseguire il codice, è necessario seguire i seguenti passaggi: 

• Installazione dei requisiti

• Importazione del dataset: il dataset breast_cancer.csv è già presente nel repository e pronto per l'uso; individuazione di valori anomali, mancanti e rimozione dei duplicati, con conseguente ottenimento dataset nuovo. 
Standardizzazione delle features numeriche del dataset; divisione del dataset in features e target label; standardizzazione delle features; salvataggio del dataset standardizzato in un nuovo file CSV.

• Sviluppo del modello: 
- Sviluppo di un classificatore KNN
- Numero di vicini (k) da usare nel classificatore KNN. Questo parametro incide su come il modello classifica i nuovi dati basandosi sui dati di addestramento.

• Valutazione del modello:
- Si può scegliere tra due metodi: Holdout e K-fold Cross Validation. Queste opzioni determinano come vengono valutate le performance del modello.

HOLDOUT: questo metodo divide il dataset in due parti, una per l'addestramento ed una per la fase di test. E' utile quando si ha un dataset di grandi dimensioni.

K-FOLD CROSS VALIDATION: questo metodo divide il dataset in k-fold e utilizza k-1 fold per l'addestramento e 1 fold per il test. E' utile quando si ha un dataset di piccole dimensioni.

- Percentuale di test: è un parametro che determina la percentuale di dati da utilizzare per il test. Questo valore è utilizzato solo se si sceglie Holdout come metodo di valutazione.
- Numero di fold: è un parametro che determina il numero di fold da utilizzare. Questo valore è utilizzato solo se si sceglie K-fold Cross Validation come metodo di valutazione.
- Metriche di valutazione: determinano il modo in cui vengono valutate le performance del modello. Si hanno le seguenti metriche:
  • Accuracy Rate: metrica che indica quante volte il nostro modello ha correttamente classificato un item nel nel nostro dataset rispetto al totale.
  • Error Rate: metrica che indica quante volte il nostro modello ha erroneamente classificato correttamente un item nel nel nostro dataset rispetto al totale
  • Sensitivity: metrica che indica la capacità del modello di individuare i casi positivi correttamente.
  • Specificity: metrica che indica la capacità del modello di individuare i casi negativi corretamente.
  • Geometric Mean: misura l'equilibrio tra Sensibilità e Specificità.

- Esecuzione del programma: il programma è eseguibile tramite il file main.py, specificando le opzioni di input come argomenti della linea di comando, quando e nella modalità in cui richiesto.

