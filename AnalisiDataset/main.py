import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore


# Caricamento del dataset
def load_data(filepath):
    return pd.read_csv(filepath)


# 1. Verifica se il dataset è bilanciato
def check_class_balance(df, target_column):
    class_distribution = df[target_column].value_counts()
    print("\nDistribuzione delle classi:")
    print(class_distribution)

    # Se la differenza tra la classe più piccola e la più grande è significativa, il dataset non è bilanciato
    balance_ratio = class_distribution.min() / class_distribution.max()
    if balance_ratio > 0.4:
        print("\nIl dataset è bilanciato.")
    else:
        print("\nIl dataset non è bilanciato.")


# 2. Verifica se ci sono valori mancanti
def check_missing_values(df):
    missing_data = df.isnull().sum()
    print("\nValori mancanti per ogni colonna:")
    print(missing_data[missing_data > 0])

    if missing_data.sum() > 0:
        print("\nCi sono valori mancanti nel dataset.")
    else:
        print("\nNon ci sono valori mancanti nel dataset.")


# 3. Identificazione degli outlier
def check_outliers(df):
    # Rimuovi la colonna 'ID' prima di calcolare gli outlier
    df = df.drop(columns=['ID'], errors='ignore')  # 'errors="ignore"' evita errori se 'ID' non esiste
    numeric_features = df.select_dtypes(include=[np.number])

    # Calcola gli Z-score
    z_scores = numeric_features.apply(zscore)
    outliers = (np.abs(z_scores) > 3).sum()  # Consideriamo outlier quelli con Z-score > 3

    print("\nNumero di outlier per feature (Z-score > 3):")
    print(outliers)


# Funzione main che esegue tutte le analisi
def main():
    filepath = 'Breast_Wisconsin_Dataset.csv'
    target_column = 'Diagnosis'

    # Carica il dataset
    df = load_data(filepath)

    # Esegui le verifiche
    check_class_balance(df, target_column)
    check_missing_values(df)
    check_outliers(df)


# Esegui il programma
if __name__ == "__main__":
    main()
