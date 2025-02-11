import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BreastCancerCSVLoader:
    def __init__(self, file_path="Breast_Wisconsin_Dataset.csv", target_column="Diagnosis", scale_data=True):
        """
        Inizializza il caricatore del dataset dal file CSV.

        :param file_path: Percorso del file CSV. (default "Breast_Wisconsin_Dataset.csv")
        :param target_column: Nome della colonna target (default "Diagnosis").
        :param scale_data: Se True, applica la standardizzazione (default True).
        """
        self.file_path = file_path
        self.target_column = target_column
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        self.load_data()

    def load_data(self):
        """Carica il dataset dal CSV e prepara le feature e il target."""
        df = pd.read_csv(self.file_path)

        # Rimuove colonne inutili (come ID)
        df = df.drop(columns=[col for col in ['id', 'Unnamed: 32'] if col in df.columns], errors='ignore')

        # Converte la colonna target in numerico
        if df[self.target_column].dtype == 'object':
            df[self.target_column] = df[self.target_column].map({'M': 1, 'B': 0})  # M=Maligno, B=Benigno

        # Separa features e target
        self.feature_names = [col for col in df.columns if col != self.target_column]
        self.target_names = ['Benigno', 'Maligno']
        self.X = df[self.feature_names].values
        self.y = df[self.target_column].values

        # Standardizzazione se richiesta
        if self.scale_data:
            self.X = self.standardize(self.X)  # Applica la standardizzazione

    def standardize(self, X):
        """Applica la standardizzazione alle features."""
        return self.scaler.fit_transform(X) if self.scaler else X

    def get_data(self, test_size=0.2, random_state=42):
        """
        Restituisce i dati divisi in training e test set.

        :param test_size: Percentuale di dati da utilizzare per il test (default 0.2).
        :param random_state: Semina casuale per la riproducibilità (default 42).
        :return: X_train, X_test, y_train, y_test
        """
        # Divisione in training e test set
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y)

    def get_feature_names(self):
        """Restituisce i nomi delle feature."""
        return self.feature_names

    def get_target_names(self):
        """Restituisce i nomi delle classi target."""
        return self.target_names

    def describe(self):
        """Restituisce un riassunto statistico del dataset."""
        df = pd.read_csv(self.file_path)
        return df.describe()


