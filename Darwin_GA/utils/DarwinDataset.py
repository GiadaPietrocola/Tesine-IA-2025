import pandas as pd
import numpy as np
from typing import Tuple


class DarwinDataset:
    def __init__(self):
        """
        Initializes the DarwinDataset object with the path to the dataset.

        :param data_path: Path to the DARWIN dataset CSV file.
        """
        self.data_path = "darwin.csv"
        self.X = None  # Feature matrix
        self.y = None  # Target (label)
        self._load_data()

    def _load_data(self):
        """Loads the dataset from the CSV file and separates the features from the target."""
        # Carica il dataset
        data = pd.read_csv(self.data_path)

        # Gestione dei valori mancanti
        data = self.handle_missing_values(data)

        # Separa le feature e il target
        self.X = data.drop(columns=["class", "ID"]).values  # Rimuove le colonne "class" e "id"
        self.y = data["class"].values  # La colonna "class" è il target
        # Nomi delle feature
        self.feature_names = data.drop(columns=["class", "ID"]).columns.values

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values in the dataset.
        Replaces missing values with the mean of the column.

        :param data: DataFrame containing the dataset.
        :return: DataFrame with handled missing values.
        """
        # Calcola la media per ogni colonna (escludendo la colonna "class" e "ID")
        col_means = data.drop(columns=["class", "ID"]).mean()

        # Sostituisce i missing values con la media della colonna
        data_filled = data.fillna(col_means)

        return data_filled

    def get_features(self) -> np.ndarray:
        """
        Returns the feature matrix.

        :return: Feature matrix (ndarray).
        """
        return self.X

    def get_target(self) -> np.ndarray:
        """
        Returns the target (label).

        :return: Target vector (ndarray).
        """
        return self.y

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns both features and target.

        :return: Tuple of numpy arrays (features, target).
        """
        return self.X, self.y

    def get_number_of_features(self) -> int:
        """
        Returns the number of features in the dataset.

        :return: Number of features.
        """
        return self.X.shape[1]

    def get_number_of_samples(self) -> int:
        """
        Returns the number of samples in the dataset.

        :return: Number of samples.
        """
        return self.X.shape[0]

    def get_feature_names(self):
        """Returns the names of the features."""
        return self.feature_names
