import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, learning_curve
from utils.BreastCancerCSVLoader import BreastCancerCSVLoader
import time
import seaborn as sns

class ModelEvaluator:
    def __init__(self, data_loader, n_runs=30, cv_folds=5):
        """
        Initializes the ModelEvaluator class to evaluate a neural network model.

        Parameters:
            data_loader (object): An instance of the data loader class that provides the dataset.
            n_runs (int): The number of runs to perform (default is 30).
            cv_folds (int): The number of folds for cross-validation (default is 5).
        """
        self.model = None
        self.data_loader = data_loader
        self.n_runs = n_runs
        self.cv_folds = cv_folds

        # Initialize lists to store results for each run
        self.accuracies = []
        self.train_scores_all = []
        self.test_scores_all = []
        self.conf_matrices = []
        self.fpr_train_all = []
        self.tpr_train_all = []
        self.roc_auc_train_all = []
        self.fpr_test_all = []
        self.tpr_test_all = []
        self.roc_auc_test_all = []
        self.cv_scores = []

    def run(self):
        """
        Runs the evaluation process across multiple runs, performs training,
        and collects results such as accuracy, ROC curves, and confusion matrices.

        Returns:
            dict: A dictionary containing aggregated results.
        """
        mean_fpr = np.linspace(0, 1, 100)
        tprs_train, tprs_test = [], []

        # Load dataset and split into train and test sets
        X_train, X_test, y_train, y_test = self.data_loader.get_data(random_state=42)

        # Initialize the model
        self.model = MLPClassifier(
            hidden_layer_sizes=(100,50),
            activation='tanh',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            shuffle=True,
            random_state=42
        )

        # Perform cross-validation
        self.cv_scores = self.cross_validate(X_train, y_train)

        # Run the training and evaluation process for each run
        for run in range(1, self.n_runs + 1):
            print(f"\nRun {run} in progress...")
            self.model = MLPClassifier(
                hidden_layer_sizes=(100,50),
                activation='tanh',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                shuffle=True,
                random_state=42+run
            )

            # Reload dataset with different random state for each run
            X_train, X_test, y_train, y_test = self.data_loader.get_data(random_state=42+run)

            # Train model
            self.train_model(X_train, y_train)

            # Evaluate model
            accuracy, train_sizes, train_mean_run, test_mean_run = self.evaluate_model(X_train, y_train, X_test, y_test)

            # Store results for this run
            self.accuracies.append(accuracy)
            self.train_scores_all.append(train_mean_run)
            self.test_scores_all.append(test_mean_run)
            self.save_confidence_scores(X_test, y_test, run)

            # Evaluate ROC and AUC for this run
            fpr_train, tpr_train, fpr_test, tpr_test, roc_auc_train, roc_auc_test = self.evaluate_roc_auc(X_train, y_train, X_test, y_test)

            # Interpolate TPR for smoother ROC curve
            tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            tprs_test.append(np.interp(mean_fpr, fpr_test, tpr_test))

            # Ensure the first point of the ROC curve is at (0,0)
            tprs_train[-1][0] = 0.0
            tprs_test[-1][0] = 0.0

        # Return aggregated results after completing all runs
        return self.aggregate_results(train_sizes, mean_fpr, tprs_train, tprs_test)

    def train_model(self, X_train, y_train):
        """
        Trains the model on the training data.

        Parameters:
            X_train (ndarray): The feature matrix for training.
            y_train (ndarray): The target vector for training.
        """
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")

    def evaluate_model(self, X_train, y_train, X_test, y_test):
        """
        Evaluates the model on test data and computes accuracy, learning curve, and confusion matrix.

        Parameters:
            X_train (ndarray): The training feature matrix.
            y_train (ndarray): The training target vector.
            X_test (ndarray): The test feature matrix.
            y_test (ndarray): The test target vector.

        Returns:
            tuple: Contains accuracy, learning curve data, and confusion matrix.
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Run completed. Accuracy: {accuracy:.4f}.")

        # Learning curve computation
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_train, y_train,
            train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
            scoring='accuracy'
        )

        train_mean_run = np.mean(train_scores, axis=1)
        test_mean_run = np.mean(test_scores, axis=1)

        # Store confusion matrix for this run
        conf_matrix = confusion_matrix(y_test, y_pred)
        self.conf_matrices.append(conf_matrix)

        return accuracy, train_sizes, train_mean_run, test_mean_run

    def evaluate_roc_auc(self, X_train, y_train, X_test, y_test):
        """
        Evaluates the model's ROC curve and AUC score.

        Parameters:
            X_train (ndarray): The training feature matrix.
            y_train (ndarray): The training target vector.
            X_test (ndarray): The test feature matrix.
            y_test (ndarray): The test target vector.

        Returns:
            tuple: Contains FPR, TPR, and AUC for both training and testing sets.
        """
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_test_proba = self.model.predict_proba(X_test)[:, 1]

        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

        roc_auc_train = auc(fpr_train, tpr_train)
        roc_auc_test = auc(fpr_test, tpr_test)

        return fpr_train, tpr_train, fpr_test, tpr_test, roc_auc_train, roc_auc_test

    def cross_validate(self, X_train, y_train):
        """
        Performs cross-validation on the training data and computes accuracy for each fold.

        Parameters:
            X_train (ndarray): The training feature matrix.
            y_train (ndarray): The training target vector.

        Returns:
            list: A list of accuracy scores for each cross-validation fold.
        """
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = []
        for train_index, test_index in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
            self.model.fit(X_train_fold, y_train_fold)
            y_pred = self.model.predict(X_test_fold)
            accuracy = np.mean(y_pred == y_test_fold)
            scores.append(accuracy)
        return scores

    def aggregate_results(self, train_sizes, mean_fpr, tprs_train, tprs_test):
        """
        Aggregates and displays the results across multiple runs.

        Parameters:
            train_sizes (array): The sizes of the training set for the learning curve.
            mean_fpr (array): The mean false positive rate for ROC.
            tprs_train (list): The true positive rate for training data.
            tprs_test (list): The true positive rate for test data.

        Returns:
            dict: A dictionary containing aggregated results.
        """
        # Compute and display overall accuracy
        mean_accuracy = np.mean(self.accuracies)
        print(f"\nAverage accuracy over {self.n_runs} runs: {mean_accuracy:.4f}")

        # Convert the results into numpy arrays for easier handling
        train_scores_all = np.array(self.train_scores_all)
        test_scores_all = np.array(self.test_scores_all)

        # Plot learning curve
        self.plot_learning_curve(train_sizes, train_scores_all, test_scores_all)

        # Plot confusion matrix
        self.plot_confusion_matrix()

        # Plot ROC curve
        self.plot_roc_curve(mean_fpr, tprs_train, tprs_test)

        # Plot cross-validation scores
        self.plot_cv_scores()

        # Plot accuracy distribution
        self.plot_accuracy_distribution()

        # Return all results in a dictionary
        return {
            "mean_accuracy": mean_accuracy,
            "train_scores_all": train_scores_all,
            "test_scores_all": test_scores_all,
            "conf_matrices": self.conf_matrices,
            "roc_auc_train_all": self.roc_auc_train_all,
            "roc_auc_test_all": self.roc_auc_test_all,
        }

    def plot_learning_curve(self, train_sizes, train_scores_all, test_scores_all):
        """
        Plots the learning curve, showing training and validation accuracy over different training set sizes.

        Parameters:
            train_sizes (array): The sizes of the training set.
            train_scores_all (ndarray): Training accuracy scores for all runs.
            test_scores_all (ndarray): Validation accuracy scores for all runs.
        """
        train_mean = np.mean(train_scores_all, axis=0)
        train_std = np.std(train_scores_all, axis=0)
        test_mean = np.mean(test_scores_all, axis=0)
        test_std = np.std(test_scores_all, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Accuracy")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
        plt.plot(train_sizes, test_mean, 'o-', color="green", label="Validation Accuracy")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="green")
        plt.title("Learning Curve")
        plt.xlabel("Training Size")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix for the final model after all runs.
        """
        mean_conf_matrix = np.mean(self.conf_matrices, axis=0)
        plt.figure(figsize=(6, 6))
        sns.heatmap(mean_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Mean Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def plot_roc_curve(self, mean_fpr, tprs_train, tprs_test):
        """
        Plots the ROC curve for both training and test sets, averaged over multiple runs.

        Parameters:
            mean_fpr (array): The mean false positive rate.
            tprs_train (list): The true positive rates for the training set.
            tprs_test (list): The true positive rates for the test set.
        """
        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_tpr_test = np.mean(tprs_test, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(mean_fpr, mean_tpr_train, color="blue", label="Train ROC curve")
        plt.plot(mean_fpr, mean_tpr_test, color="green", label="Test ROC curve")
        plt.title("Mean ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def plot_cv_scores(self):
        """
        Plots the distribution of cross-validation scores.
        """
        plt.figure(figsize=(6, 6))
        sns.boxplot(data=self.cv_scores)
        plt.title("Cross-validation Scores")
        plt.ylabel("Accuracy")
        plt.show()

    def plot_accuracy_distribution(self):
        """
        Plots the distribution of accuracies across all runs.
        """
        plt.figure(figsize=(6, 6))
        sns.histplot(self.accuracies, kde=True, color="blue")
        plt.title("Accuracy Distribution")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        plt.show()


if __name__ == "__main__":

    data_loader = BreastCancerCSVLoader()
    evaluator = ModelEvaluator(data_loader)
    results = evaluator.run()
