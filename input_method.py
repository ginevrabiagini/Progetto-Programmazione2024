import pandas as pd

class Input:
    """
    Classe per gestire l'input dell'utente riguardo al dataset, ai parametri del modello e alle metriche di valutazione.
    """

    def __init__(self, path_dataset=None, k=None, evaluation=None, training=None, K=None, metrics=[], data=None, target_column=None):
        """
        Inizializza la classe Input con i parametri forniti.

        Parameters:
        path_dataset (str): Il percorso del file del dataset
        k (int): Il numero di vicini da utilizzare per il classificatore
        evaluation (int): Il metodo di valutazione scelto (1 per Holdout, 2 per KFoldCrossValidation)
        training (int): La percentuale di dati da utilizzare nel set di training
        K (int): Il numero di fold per la validazione incrociata KFold
        metrics (list): La lista delle metriche da calcolare
        data (DataFrame): Il DataFrame contenente i dati del dataset
        target_column (str): Il nome della colonna target nel dataset
        """
        self.k = k
        self.evaluation = evaluation
        self.training = training
        self.K = K
        self.metrics = metrics
        self.path_dataset = path_dataset
        self.data = data
        self.target_column = target_column

    def get_path(self):
        """
        Chiede all'utente di inserire il percorso del dataset e tenta di caricarlo

        Stampa un messaggio di successo o un messaggio di errore se il file non viene trovato
        """
        while True:
            self.path_dataset = input("Inserisci il path assoluto del tuo dataset: ")
            if self.path_dataset.startswith('"') and self.path_dataset.endswith('"'):
                self.path_dataset = self.path_dataset[1:-1]  # Rimuove i doppi apici
            try:
                self.data = pd.read_csv(self.path_dataset)
                print("Dataset caricato con successo.")
                break
            except FileNotFoundError:
                print(f"Il file '{self.path_dataset}' non è stato trovato.")

    def get_k(self):
        """
        Chiede all'utente di inserire il numero di vicini da utilizzare per il classificatore KNN

        Stampa un messaggio di errore se l'input non è un numero intero valido
        """
        while True:
            try:
                self.k = int(input("Inserisci il numero di vicini da utilizzare per il classificatore: "))
                break
            except ValueError:
                print("Errore: Inserisci un numero intero valido.")

    def get_evaluation_method(self):
        """
        Chiede all'utente di scegliere il metodo di valutazione del modello (Holdout o KFoldCrossValidation)

        Stampa il metodo di valutazione scelto
        """
        print("Scegli come valutare il modello:")
        print("1. Holdout")
        print("2. KFoldCrossValidation")
        while True:
            try:
                choice = int(input("Inserisci il numero corrispondente alla tua scelta: "))
                if choice == 1:
                    self.evaluation = 1
                    break
                elif choice == 2:
                    self.evaluation = 2
                    self.get_K()
                    break
                else:
                    print("Errore: Inserisci un numero valido.")
            except ValueError:
                print("Errore: Inserisci un numero intero valido.")

    def get_training_percentage(self):
        """
        Chiede all'utente di inserire la percentuale di dati da utilizzare nel set di training

        Stampa un messaggio di errore se l'input non è un numero intero valido o non è compreso tra 0 e 100
        """
        while True:
            try:
                self.training = int(input("Inserisci la percentuale di dati da utilizzare nel set di training (0-100): "))
                if 0 < self.training < 100:
                    break
                else:
                    print("Errore: Inserisci un numero intero valido compreso tra 0 e 100.")
            except ValueError:
                print("Errore: Inserisci un numero intero valido.")

    def get_K(self):
        """
        Chiede all'utente di inserire il numero di fold per la validazione incrociata KFold

        Stampa un messaggio di errore se l'input non è un numero intero valido o non è maggiore di 0
        """
        while True:
            try:
                self.K = int(input("Inserisci il numero di esperimenti K: "))
                if self.K > 0:
                    break
                else:
                    print("Errore: Inserisci un numero intero valido maggiore di 0.")
            except ValueError:
                print("Errore: Inserisci un numero intero valido.")

    def choose_metrics(self):
        """
        Chiede all'utente di scegliere le metriche da calcolare per la valutazione del modello

        Aggiunge le metriche scelte alla lista self.metrics
        """
        print("Scegli quali metriche devono essere validate:")
        print("1. Accuracy Rate")
        print("2. Error Rate")
        print("3. Sensitivity")
        print("4. Specificity")
        print("5. Geometric Mean")
        print("6. Tutte le metriche disponibili")
        while True:
            choices = input("Inserisci i numeri corrispondenti alle metriche separate da uno spazio: ")
            choices = choices.split()
            valid_choices = ["1", "2", "3", "4", "5", "6"]
            if all(choice in valid_choices for choice in choices):
                break
            else:
                print("Errore: Inserisci numeri validi.")

        for choice in choices:
            if choice == "1":
                self.metrics.append(1)
            elif choice == "2":
                self.metrics.append(2)
            elif choice == "3":
                self.metrics.append(3)
            elif choice == "4":
                self.metrics.append(4)
            elif choice == "5":
                self.metrics.append(5)
            elif choice == "6":
                self.metrics = [1, 2, 3, 4, 5]
                break

    def get_target_column(self):
        """
        Chiede all'utente di inserire il nome della colonna target

        Verifica se la colonna esiste nel dataset. Stampa un messaggio di errore se la colonna non esiste
        """
        while True:
            target_column = input("Inserisci il nome della colonna target(Class) : ")
            if target_column in self.data.columns:
                self.target_column = target_column
                break
            else:
                print(f"Errore: La colonna '{target_column}' non esiste nel dataset.")

    def get_input(self):
        """
        Chiede all'utente di fornire tutti gli input necessari:
        - Percorso del dataset
        - Numero di vicini per KNN
        - Percentuale di training set
        - Metodo di valutazione
        - Metriche da calcolare
        - Nome della colonna target

        Returns:
        self: L'istanza della classe Input con i valori forniti dall'utente
        """
        self.get_path()
        self.get_k()
        self.get_training_percentage()
        self.get_evaluation_method()
        self.choose_metrics()
        self.get_target_column()
        return self
