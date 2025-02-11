import pandas as pd
import numpy as np
from typing import Tuple


class DarwinDataset:
    def __init__(self):
        """
        Inizializza l'oggetto DarwinDataset con il percorso del dataset.

        :param data_path: Percorso del file CSV del dataset DARWIN.
        """
        self.data_path = "darwin.csv"
        self.X = None  # Matrice delle feature
        self.y = None  # Target (etichetta)
        self._load_data()

    def _load_data(self):
        """Carica il dataset dal file CSV e separa le feature dal target."""
        # Carica il dataset
        data = pd.read_csv(self.data_path)

        # Gestione dei valori mancanti
        data = self.handle_missing_values(data)

        # Separa le feature e il target
        self.X = data.drop(columns=["class", "ID"]).values  # Rimuove le colonne "class" e "id""
        self.y = data["class"].values  # La colonna "class" è il target
        # Nomi delle feature
        self.feature_names = data.drop(columns=["class", "ID"]).columns.values

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gestisce i valori mancanti nel dataset.
        Sostituisce i valori mancanti con la media della colonna.

        :param data: DataFrame contenente il dataset.
        :return: DataFrame con i valori mancanti gestiti.
        """
        # Calcola la media per ogni colonna (escludendo la colonna "class" e "ID")
        col_means = data.drop(columns=["class", "ID"]).mean()

        # Sostituisce i missing values con la media della colonna
        data_filled = data.fillna(col_means)

        return data_filled

    def get_features(self) -> np.ndarray:
        """
        Restituisce la matrice delle feature.

        :return: Matrice (ndarray) delle feature.
        """
        return self.X

    def get_target(self) -> np.ndarray:
        """
        Restituisce il target (etichetta).

        :return: Vettore (ndarray) del target.
        """
        return self.y

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restituisce sia le feature che il target.

        :return: Tuple di numpy array (features, target).
        """
        return self.X, self.y

    def get_number_of_features(self) -> int:
        """
        Restituisce il numero di feature nel dataset.

        :return: Numero di feature.
        """
        return self.X.shape[1]

    def get_number_of_samples(self) -> int:
        """
        Restituisce il numero di campioni nel dataset.

        :return: Numero di campioni.
        """
        return self.X.shape[0]

    def get_feature_names(self):
        """Restituisce i nomi delle feature."""
        return self.feature_names
