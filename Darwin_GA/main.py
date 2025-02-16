import sys

import matplotlib
import matplotlib.pyplot as plt

from utils.DarwinDataset import DarwinDataset

sys.dont_write_bytecode = True

from src.genetic_algorithm import GeneticAlgorithm
import numpy as np
from typing import List, Tuple
from scipy.stats import pointbiserialr
import seaborn as sns


def precompute_correlations(dataset: DarwinDataset):
    """
    Pre-computa le correlazioni tra le feature e il target e tra le feature stesse.
    """
    x = dataset.get_features()
    y = dataset.get_target()
    y_numeric = np.where(y == 'P', 0, 1)  # Converte il target binario in numerico

    # Correlazioni tra feature e target usando la correlazione punto-biseriali
    correlations_with_target = np.array([pointbiserialr(x[:, i], y_numeric)[0] for i in range(x.shape[1])])

    # Matrice di correlazione tra le feature
    feature_correlation_matrix = np.corrcoef(x.T)

    return correlations_with_target, feature_correlation_matrix


def fitness_function(
        chromosome: np.ndarray,
        correlations_with_target: np.ndarray,
        feature_correlation_matrix: np.ndarray
) -> float:
    """
    Funzione di fitness che utilizza correlazioni pre-computate.
    """
    # Seleziona le feature in base al cromosoma
    selected_features = np.where(chromosome == 1)[0]
    num_selected = len(selected_features)
    n_max_features = 450
    # Somma delle correlazioni assolute con il target per le feature selezionate normalizzata
    correlation_sum = np.sum(np.abs(correlations_with_target[selected_features])) / num_selected

    # Penalità per la correlazione tra le feature selezionate
    # Estrae la sotto-matrice per le feature selezionate
    selected_correlation_matrix = feature_correlation_matrix[np.ix_(selected_features, selected_features)]
    # Somma le correlazioni superiori alla diagonale (k=1) normalizzata
    inter_feature_penalty = np.sum(np.abs(np.triu(selected_correlation_matrix, k=1))) / ((num_selected * (num_selected - 1)) / 2)

    # Calcola il valore di fitness
    fitness_value = inter_feature_penalty + num_selected/n_max_features - correlation_sum

    return fitness_value

# Decodifica del Cromosoma
def decode_ga_chromosome(chromosome: np.ndarray) -> np.ndarray:
    """Decodifica il cromosoma binario in un array di feature selezionate."""
    # Ogni cromosoma ha la lunghezza uguale al numero di feature nel dataset
    return np.where(chromosome == 1)[0] # Restituisce gli indici delle feature selezionate


def run_optimization():
    """Run the selected optimization method."""
    print("Feature Selection Optimization")
    print("=" * 50)

    # Parametri
    n_runs=1
    MAX_GENERATIONS = 200
    POPULATION_SIZE = 100


    print("\nStarting Genetic Algorithm Optimization...")

    # Istanzia il dataset DARWIN
    dataset=DarwinDataset()

    # Pre-computa le correlazioni tra le feature e il target
    correlations_with_target, feature_correlation_matrix = precompute_correlations(dataset)

    feature_counts = np.zeros(dataset.get_number_of_features())
    best_fitness_values = []
    fitness_history = []
    fitness_generation = []
    feature_selection_history = []

    for run in range(1, n_runs + 1):
        print(f"\nRun {run} in progress...")

        # Parametri dell'algoritmo genetico
        ga = GeneticAlgorithm(
            population_size=POPULATION_SIZE,
            chromosome_length=dataset.get_number_of_features(),
            fitness_func=fitness_function,
            correlations_with_target=correlations_with_target,
            feature_correlation_matrix=feature_correlation_matrix,
            dataset=dataset,
            max_generations=MAX_GENERATIONS,
            mutation_rate=0.05,
            crossover_rate=0.8,
            elitism=True,
            verbose=True,
            random_state=42+run,
            stagnation_generation=10,
            tolerance=1e-4
        )

        # Esegue l'algoritmo genetico
        best_chromosome = ga.evolve()
        best_params_ga = decode_ga_chromosome(best_chromosome)
        best_fitness_ga = fitness_function(best_chromosome,correlations_with_target,feature_correlation_matrix)
        best_fitness_values.append(best_fitness_ga)

        # Visualizza i risultati
        print("\nOptimization Results")
        print("=" * 50)

        # Ottiene i nomi delle feature dal dataset
        feature_names = dataset.get_feature_names()

        # Seleziona i nomi delle feature in base agli indici 'best_params_ga'
        selected_feature_names = feature_names[best_params_ga]
        print("Selected Features: ", selected_feature_names)

        print(f"Best Fitness Value: {best_fitness_ga:.4f}")
        print(f"Selected Feature Indices: {best_params_ga}")

        selected_features = decode_ga_chromosome(best_chromosome)
        feature_counts[selected_features] += 1

        # Salva la fitness history della run corrente
        ga.save_fitness_history(fitness_history)

        # Salva le feature selezionate nella run corrente
        ga.save_feature_selection_history(feature_selection_history)

        # Salva tutte le fitness per generazione della run corrente
        ga.save_fitness_generation(fitness_generation)


    # Medie
    best_fitness_mean = np.mean([run[0] for run in fitness_history], axis=0)
    avg_fitness_mean = np.mean([run[1] for run in fitness_history], axis=0)
    avg_features_mean = np.mean(feature_selection_history, axis=0)

    print(f"\nAverage Best Fitness Value: {np.mean(best_fitness_values):.4f}")
    print("Most Selected Features:", np.argsort(feature_counts)[-10:][::-1])

    #Plots
    plot_fitness_function(max_generations=MAX_GENERATIONS, best_fitness_mean=best_fitness_mean, avg_fitness_mean=avg_fitness_mean)
    plot_feature_selection_frequency(max_generations=MAX_GENERATIONS, chromosome_length=dataset.get_number_of_features(), feature_selection_history=avg_features_mean)
    plot_boxplot(fitness_history=fitness_generation)

def plot_fitness_function(max_generations: int, best_fitness_mean: np.ndarray, avg_fitness_mean: np.ndarray):
    """
    Plots the mean best fitness and mean average fitness across multiple runs over generations.

    Parameters:
        max_generations (int): The total number of generations in the evolutionary process.
        best_fitness_mean (np.ndarray): The mean of the best fitness values across all runs, for each generation.
        avg_fitness_mean (np.ndarray): The mean of the average fitness values across all runs, for each generation.
    """

    sns.set_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Traccia l'andamento della fitness
    ax.plot(range(max_generations), best_fitness_mean, 'r-', label='Mean Best Fitness', linewidth=2)
    ax.plot(range(max_generations), avg_fitness_mean, 'b-', label='Mean Average Fitness', linewidth=2)

    # Limiti fissi per gli assi
    ax.set_xlim(0, max_generations)
    ax.set_ylim(0, 1)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Generations', fontsize=10)
    ax.set_ylabel('Fitness', fontsize=10)
    ax.set_title('Feature Selection Progress (Mean Over Runs)', fontsize=12, pad=20)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_feature_selection_frequency(max_generations: int, chromosome_length: int, feature_selection_history: np.ndarray):
    """
    Plots the frequency of feature selection across generations.

    Parameters:
        max_generations (int): The total number of generations.
        chromosome_length (int): The number of features (length of the chromosome).
        feature_selection_history (np.ndarray): A binary history of feature selection across generations,
                                                with shape (max_generations, chromosome_length), where 1 indicates
                                                the selection of a feature and 0 indicates its non-selection.
    """

    # Frequenza di selezione di ogni feature su tutte le generazioni
    feature_frequencies = feature_selection_history.sum(axis=0) / max_generations
    plt.figure(figsize=(12, 6))
    plt.bar(range(chromosome_length), feature_frequencies, color='lightblue', edgecolor='none', width=1)
    plt.title("Feature Selection Frequency")
    plt.xlabel("Feature Index")
    plt.ylabel("Selection Frequency (%)")

    plt.tight_layout()
    plt.show()

def plot_boxplot(fitness_history: List[List[np.ndarray]]):
    """
    Creates a box plot to visualize the fitness distribution across generations.
    This function handles the case where different runs have a varying number of generations.

    Parameters:
        fitness_history (List[List[np.ndarray]]): A list of runs, where each run contains an array of fitness values
                                                   for each generation.
    """
    # Dizionario per raccogliere i valori di fitness per ogni generazione
    fitness_data_dict = {}

    # Iteriamo su tutte le run e raccogliamo i fitness per ogni generazione
    for run in fitness_history:
        for gen, fitness_values in enumerate(run):  # Ogni run può avere un numero variabile di generazioni
            if gen not in fitness_data_dict:
                fitness_data_dict[gen] = []
            fitness_data_dict[gen].extend(fitness_values)  # Aggiungiamo i fitness per la generazione corrente

    # Ordiniamo le generazioni, nel caso in cui non siano tutte consecutive
    sorted_generations = sorted(fitness_data_dict.keys())

    # Creiamo una lista ordinata di dati per il box plot
    fitness_data_reshaped = [fitness_data_dict[gen] for gen in sorted_generations]

    # Creiamo il boxplot per visualizzare la distribuzione dei fitness
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=fitness_data_reshaped)
    plt.title("Box plot of Fitness Distributions")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.xticks(ticks=range(len(sorted_generations)), labels=sorted_generations, rotation=90)
    plt.show()



if __name__ == "__main__":
    run_optimization()