import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(losses, accuracies=None, window_size=50):
    """
    Plots the training metrics (loss and accuracy) over time.
    
    Parameters:
    losses (list): List of loss values during training
    accuracies (list): Optional list of accuracy values during training
    window_size (int): Window size for moving average smoothing
    """
    # Create figure with appropriate size
    if accuracies is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(figsize=(12, 5))
    
    # Calculate moving averages for smoothing
    def moving_average(data, window_size):
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode='valid')
    
    epochs = range(1, len(losses) + 1)
    smoothed_losses = moving_average(losses, window_size)
    
    # Plot training loss
    ax1.plot(epochs[window_size-1:], smoothed_losses, 
             color='#ff7f0e', label='Smoothed Loss')
    ax1.plot(epochs, losses, color='#1f77b4', alpha=0.2, label='Training Loss')
    
    ax1.set_title('Training Loss Over Time', size=14, pad=15)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot accuracy if provided
    if accuracies is not None:
        smoothed_acc = moving_average(accuracies, window_size)
        
        ax2.plot(epochs[window_size-1:], smoothed_acc, 
                color='#2ca02c', label='Smoothed Accuracy')
        ax2.plot(epochs, accuracies, color='#d62728', 
                alpha=0.2, label='Training Accuracy')
        
        ax2.set_title('Training Accuracy Over Time', size=14, pad=15)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_prediction_distribution(y_true, y_pred, threshold=0.5):
    """
    Visualizes the distribution of predictions against true values.
    
    Parameters:
    y_true (array): True binary labels
    y_pred (array): Predicted probabilities
    threshold (float): Classification threshold
    """
    plt.figure(figsize=(10, 6))
    
    # Plot prediction distributions for each class
    for label, color in zip([0, 1], ['#ff7f0e', '#2ca02c']):
        mask = y_true.flatten() == label
        plt.hist(y_pred[mask], bins=30, alpha=0.6, color=color,
                label=f'Class {label}', density=True)
    
    # Add threshold line
    plt.axvline(x=threshold, color='red', linestyle='--', 
               label='Decision Threshold')
    
    plt.title('Distribution of Predictions by True Class', size=14, pad=15)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()