# genetic_algorithm.py

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import time
from matplotlib.animation import FuncAnimation
from scipy.stats import entropy
import seaborn as sns
class GeneticAlgorithm:
    """
    A comprehensive Genetic Algorithm implementation with smooth real-time visualization
    and fixed axes for better readability.
    """

    def __init__(
            self,
            population_size: int,
            chromosome_length: int,
            fitness_func: Callable,
            correlations_with_target: np.ndarray,
            feature_correlation_matrix: np.ndarray,
            dataset: 'DarwinDataset',  # Aggiungi il parametro per il dataset
            mutation_rate: float = 0.01,
            crossover_rate: float = 0.8,
            elitism: bool = True,
            max_generations: int = 100,
            verbose: bool = True,
            animation_interval: int = 500,
            plot_pause: float = 0.5,
            random_state=42
    ):


        """Initialize the Genetic Algorithm with the given parameters."""
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.fitness_func = fitness_func
        self.dataset = dataset  # Salva il dataset come attributo
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.max_generations = max_generations
        self.verbose = verbose
        self.animation_interval = animation_interval
        self.plot_pause = plot_pause
        self.correlations_with_target = correlations_with_target
        self.feature_correlation_matrix = feature_correlation_matrix
        np.random.seed(random_state)


        # Initialize tracking variables
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.fitness_history = []
        self.feature_selection_history = np.zeros(self.chromosome_length)
        self.current_generation = 0

        # Create initial random population
        self.population = np.random.randint(2, size=(population_size, chromosome_length))

        # Create full-length arrays for data
        self.generation_points = np.arange(max_generations)
        self.best_fitness_data = np.zeros(max_generations)
        self.avg_fitness_data = np.zeros(max_generations)

    def _calculate_population_fitness(self) -> np.ndarray:
        """Calculate fitness for all individuals in the population."""
        return np.array([self.fitness_func(ind, self.correlations_with_target, self.feature_correlation_matrix) for ind in self.population])

    def _calculate_feature_selection_frequency(self) -> np.ndarray:
        """Calculate the frequency of feature selection across the population."""
        # Somma le selezioni delle feature in ogni individuo (1 per ogni feature selezionata)
        feature_selection_frequency = np.sum(self.population, axis=0)
        return feature_selection_frequency

    def _tournament_selection(self, fitness_values: np.ndarray, tournament_size: int = 3) -> np.ndarray:
        """Select individual using tournament selection."""
        tournament_indices = np.random.choice(self.population_size, size=tournament_size, replace=False)
        tournament_fitness = fitness_values[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return self.population[winner_idx]

    def _roulette_wheel_selection(self, fitness_values: np.ndarray) -> np.ndarray:
        """Select individual using roulette wheel selection."""
        inverted_fitness = 1 / (fitness_values + 1e-6)
        total_fitness = np.sum(inverted_fitness)
        selection_probs = inverted_fitness / total_fitness  # Probabilità proporzionale alla fitness
        cumulative_probs = np.cumsum(selection_probs)  # Probabilità cumulative

        # Genera un numero casuale tra 0 e 1 per selezionare un individuo
        rand = np.random.random()

        # Trova l'individuo corrispondente alla probabilità cumulativa
        selected_idx = np.where(cumulative_probs >= rand)[0][0]
        return self.population[selected_idx]

    def _select_parents(self, fitness_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select two parents using either tournament or roulette wheel selection."""
        if np.random.random() < 0.5:  # 50% probabilità di usare il torneo
            parent1 = self._tournament_selection(fitness_values)
            parent2 = self._tournament_selection(fitness_values)
        else:  # Altrimenti, utilizziamo la ruota della roulette
            parent1 = self._roulette_wheel_selection(fitness_values)
            parent2 = self._roulette_wheel_selection(fitness_values)

        return parent1, parent2


    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform two-point crossover between parents."""
        if np.random.random() < self.crossover_rate:
            points = sorted(np.random.choice(self.chromosome_length - 1, size=2, replace=False) + 1)
            offspring1 = np.concatenate([
                parent1[:points[0]],
                parent2[points[0]:points[1]],
                parent1[points[1]:]
            ])
            offspring2 = np.concatenate([
                parent2[:points[0]],
                parent1[points[0]:points[1]],
                parent2[points[1]:]
            ])
            return offspring1, offspring2
        return parent1.copy(), parent2.copy()

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Perform bit-flip mutation on the chromosome."""
        mutation_mask = np.random.random(self.chromosome_length) < self.mutation_rate
        if mutation_mask.any():
            chromosome = chromosome.copy()
            chromosome[mutation_mask] = 1 - chromosome[mutation_mask]
        return chromosome

    def save_fitness_history(self, history: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Save the best and average fitness per generation for later analysis.

        Parameters:
            history (List[Tuple[np.ndarray, np.ndarray]]): Lista in cui salvare i risultati delle run.
        """
        history.append((self.best_fitness_data, self.avg_fitness_data))

    def save_fitness_generation(self, history: List[List[float]]):
        """
        Save the fitness values per generation for later analysis.

        Parameters:
            history (List[List[float]]): Lista in cui salvare i risultati delle run.
        """
        history.append(self.fitness_history)

    def save_feature_selection_history(self, history: List[np.ndarray]):
        """
        Save the frequency of feature selection per generation for later analysis.

        Parameters:
            history (List[np.ndarray]): Lista per salvare la frequenza di selezione delle feature.
        """
        history.append(self.feature_selection_history)

    def calculate_population_diversity(self):
        """Calculate diversity using average Hamming distance."""
        num_individuals = len(self.population)
        total_hamming_distance = 0
        comparisons = 0

        for i in range(num_individuals):
            for j in range(i + 1, num_individuals):
                hamming_distance = np.sum(self.population[i] != self.population[j])
                total_hamming_distance += hamming_distance
                comparisons += 1

        return total_hamming_distance / (comparisons * len(self.population[0]))

    def evolve(self) -> np.ndarray:
        """Run the genetic algorithm evolution process."""

        start_time = time.time()
        max_time = 200  # Tempo massimo in secondi per ogni run

        for generation in range(self.max_generations):
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time:
                print(f"Max time reached: {elapsed_time:.2f} seconds.")
                break

            self.current_generation = generation

            # Calculate fitness
            fitness_values = self._calculate_population_fitness()
            best_fitness = np.min(fitness_values)
            avg_fitness = np.mean(fitness_values)
            self.fitness_history.append(fitness_values)

            # Store fitness history
            self.best_fitness_data[generation] = best_fitness
            self.avg_fitness_data[generation] = avg_fitness

            self.feature_selection_history = self._calculate_feature_selection_frequency()

            if self.verbose:
                print(f"\nGeneration {generation + 1}/{self.max_generations}")
                print(f"Best Fitness: {best_fitness:,.2f}")
                print(f"Average Fitness: {avg_fitness:,.2f}")
                print(f"Population Diversity: {self.calculate_population_diversity():,.2f}")

            # Create new population
            new_population = []

            # Elitism
            if self.elitism:
                elite_idx = np.argmin(fitness_values)
                new_population.append(self.population[elite_idx].copy())

            # Create rest of new population
            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents(fitness_values)
                offspring1, offspring2 = self._crossover(parent1, parent2)
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)
                new_population.extend([offspring1, offspring2])

            # Update population
            new_population = new_population[:self.population_size]
            self.population = np.array(new_population)


        plt.ioff()  # Turn off interactive mode
        final_fitness = self._calculate_population_fitness()
        best_idx = np.argmin(final_fitness)

        print(f"Evolution completed in {elapsed_time:.2f}s")
        return self.population[best_idx]



