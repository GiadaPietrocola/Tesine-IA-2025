import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BreastCancerCSVLoader:
    def __init__(self, file_path="Breast_Wisconsin_Dataset.csv", target_column="Diagnosis", scale_data=True):
        """
        Initializes the dataset loader from the CSV file.

        Parameters:
            file_path (str): Path to the CSV file (default is "Breast_Wisconsin_Dataset.csv").
            target_column (str): Name of the target column (default is "Diagnosis").
            scale_data (bool): If True, applies standardization (default is True).
        """
        self.file_path = file_path
        self.target_column = target_column
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        self.load_data()

    def load_data(self):
        """Loads the dataset from the CSV file and prepares features and target."""
        df = pd.read_csv(self.file_path)

        # Removes unnecessary columns, such as ID and others
        df = df.drop(columns=[col for col in ['id', 'Unnamed: 32'] if col in df.columns], errors='ignore')

        # Converts the target column to numeric format: 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
        if df[self.target_column].dtype == 'object':
            df[self.target_column] = df[self.target_column].map({'M': 1, 'B': 0})

        # Splits the dataset into features (X) and target (y)
        self.feature_names = [col for col in df.columns if col != self.target_column]
        self.target_names = ['Benign', 'Malignant']
        self.X = df[self.feature_names].values  # Loads the features
        self.y = df[self.target_column].values  # Loads the target

        # Applies standardization if requested
        if self.scale_data:
            self.X = self.standardize(self.X)

    def standardize(self, X):
        """Applies standardization (scaling) to the features."""
        return self.scaler.fit_transform(X) if self.scaler else X

    def get_data(self, test_size=0.2, random_state=42):
        """
        Returns the data split into training and test sets.

        Parameters:
            test_size (float): Percentage of data to use for testing (default is 0.2).
            random_state (int): Random seed for reproducibility (default is 42).

        Returns:
            tuple: X_train, X_test, y_train, y_test (training and test sets).
        """
        # Splits the dataset into training and test sets
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y)

    def get_feature_names(self):
        """Returns the names of the dataset features."""
        return self.feature_names

    def get_target_names(self):
        """Returns the names of the target classes (Benign, Malignant)."""
        return self.target_names

    def describe(self):
        """Returns a statistical summary of the dataset."""
        df = pd.read_csv(self.file_path)
        return df.describe()
