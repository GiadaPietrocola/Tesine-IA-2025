import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, learning_curve
from utils.BreastCancerCSVLoader import BreastCancerCSVLoader
import time
import seaborn as sns

class ModelEvaluator:
    def __init__(self, data_loader, n_runs=30, cv_folds=5):
        self.model = None
        self.data_loader = data_loader
        self.n_runs = n_runs
        self.cv_folds = cv_folds

        # Initialize lists to store results
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
        mean_fpr = np.linspace(0, 1, 100)
        tprs_train, tprs_test = [], []

        X_train, X_test, y_train, y_test = self.data_loader.get_data(random_state=42)

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

        # Cross-validation scores
        self.cv_scores = self.cross_validate(X_train, y_train)

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


            X_train, X_test, y_train, y_test = self.data_loader.get_data(random_state=42+run)

            # Train model
            self.train_model(X_train, y_train)

            # Evaluate model
            accuracy, train_sizes, train_mean_run, test_mean_run = self.evaluate_model(X_train, y_train, X_test, y_test)

            # Store results
            self.accuracies.append(accuracy)
            self.train_scores_all.append(train_mean_run)
            self.test_scores_all.append(test_mean_run)
            self.save_confidence_scores(X_test, y_test, run)

            # ROC and AUC
            fpr_train, tpr_train, fpr_test, tpr_test, roc_auc_train, roc_auc_test = self.evaluate_roc_auc(X_train, y_train, X_test, y_test)

            tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            tprs_test.append(np.interp(mean_fpr, fpr_test, tpr_test))

            tprs_train[-1][0] = 0.0
            tprs_test[-1][0] = 0.0

        # Return all results after the loop
        return self.aggregate_results(train_sizes, mean_fpr, tprs_train, tprs_test)

    def train_model(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")

    def evaluate_model(self, X_train, y_train, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Run completed. Accuracy: {accuracy:.4f}.")

        # Learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_train, y_train,
            train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
            scoring='accuracy'
        )

        train_mean_run = np.mean(train_scores, axis=1)
        test_mean_run = np.mean(test_scores, axis=1)

        conf_matrix = confusion_matrix(y_test, y_pred)
        self.conf_matrices.append(conf_matrix)

        return accuracy, train_sizes, train_mean_run, test_mean_run

    def evaluate_roc_auc(self, X_train, y_train, X_test, y_test):
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_test_proba = self.model.predict_proba(X_test)[:, 1]

        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

        roc_auc_train = auc(fpr_train, tpr_train)
        roc_auc_test = auc(fpr_test, tpr_test)

        return fpr_train, tpr_train, fpr_test, tpr_test, roc_auc_train, roc_auc_test

    def cross_validate(self, X_train, y_train):
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
        # Compute and display overall results
        mean_accuracy = np.mean(self.accuracies)
        print(f"\nAverage accuracy over {self.n_runs} runs: {mean_accuracy:.4f}")

        # Compute learning curves and plot
        train_scores_all = np.array(self.train_scores_all)
        test_scores_all = np.array(self.test_scores_all)

        # Plot learning curve
        self.plot_learning_curve(train_sizes,train_scores_all, test_scores_all)

        # Confusion matrix
        self.plot_confusion_matrix()

        # ROC curve
        self.plot_roc_curve(mean_fpr, tprs_train, tprs_test)

        # Cross-validation score plot
        self.plot_cv_scores()

        self.plot_accuracy_distribution()


        return {
            "mean_accuracy": mean_accuracy,
            "train_scores_all": train_scores_all,
            "test_scores_all": test_scores_all,
            "conf_matrices": self.conf_matrices,
            "roc_auc_train_all": self.roc_auc_train_all,
            "roc_auc_test_all": self.roc_auc_test_all,
        }

    def plot_learning_curve(self, train_sizes, train_scores_all, test_scores_all):
        train_mean = np.mean(train_scores_all, axis=0)
        train_std = np.std(train_scores_all, axis=0)
        test_mean = np.mean(test_scores_all, axis=0)
        test_std = np.std(test_scores_all, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Accuracy")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
        plt.plot(train_sizes, test_mean, 'o-', color="orange", label="Validation Accuracy")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="red")
        plt.title(f"Learning Curve (Training vs Validation Accuracy) - {self.n_runs} Runs")
        plt.xlabel("Number of Training Examples")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.grid(True, linestyle=':', color='gray', alpha=0.5)
        plt.show()

    def plot_confusion_matrix(self):
        mean_conf_matrix = np.mean(self.conf_matrices, axis=0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(mean_conf_matrix, annot=True, fmt=".2f", cmap="Blues")
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, mean_fpr, tprs_train, tprs_test):
        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_tpr_test = np.mean(tprs_test, axis=0)
        mean_auc_train = auc(mean_fpr, mean_tpr_train)
        mean_auc_test = auc(mean_fpr, mean_tpr_test)

        plt.figure(figsize=(8, 6))
        plt.plot(mean_fpr, mean_tpr_train, color='blue', label=f'Training ROC (AUC = {mean_auc_train:.2f})')
        plt.plot(mean_fpr, mean_tpr_test, color='orange', label=f'Test ROC (AUC = {mean_auc_test:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Classifier")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Mean ROC Curve over Multiple Runs')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', color='gray', alpha=0.5)
        plt.show()

    def plot_cv_scores(self):

        plt.figure(figsize=(8, 6))
        plt.bar(range(1, self.cv_folds + 1), self.cv_scores, yerr=np.std(self.cv_scores), color='blue', alpha=0.7, capsize=5)
        plt.axhline(np.mean(self.cv_scores), color='red', linestyle='dashed', label=f'Media: {np.mean(self.cv_scores):.4f}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('Distribuzione degli score in Cross-Validation')
        plt.xticks(range(1, self.cv_folds + 1))
        plt.legend()
        plt.grid(True, linestyle=':', color='gray', alpha=0.5)
        plt.show()

    def plot_accuracy_distribution(self):
        mean_acc = np.mean(self.accuracies)
        std_acc = np.std(self.accuracies)

        # Creiamo un array di indici (run)
        runs = np.arange(1, len(self.accuracies) + 1)

        # Interpolazione dei dati
        runs_fine = np.linspace(1, len(self.accuracies), 500)  # più punti per una linea liscia
        accuracies_interpolated = np.interp(runs_fine, runs, self.accuracies)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(runs_fine, accuracies_interpolated, label='Interpolazione', color='b', linewidth=2)
        plt.scatter(runs, self.accuracies, color='blue', label='Punti dati', zorder=5)  # Punti originali
        plt.title("Accuratezza Media per Run con Interpolazione")
        plt.xlabel("Run")
        plt.ylabel("Accuracy Media")
        plt.legend()
        plt.grid(True, linestyle=':', color='gray', alpha=0.5)
        plt.tight_layout()
        plt.show()


    def save_confidence_scores(self, X_test, y_test, run):
        probabilities = self.model.predict_proba(X_test)
        df = pd.DataFrame(probabilities, columns=[f'Class_{i}' for i in range(probabilities.shape[1])])

        df['Predicted_Class'] = self.model.predict(X_test)
        df['True_Label'] = y_test

        filename=f"confidence_scores_run{run}.csv"
        df.to_csv(filename, index=False, float_format='%.10f', sep=';')
        print(f"Confidence scores salvati in 'confidence_scores_run{run}.csv'.")


if __name__ == "__main__":
    # Define your model and data loader


    data_loader = BreastCancerCSVLoader()
    evaluator = ModelEvaluator(data_loader)
    results = evaluator.run()
