import networkx as nx
import matplotlib.pyplot as plt
from typing import List


def visualize_neural_network(input_size: int, hidden_layers: List[int], output_size: int = 1):
    """
    Creates a visualization of the neural network architecture with multiple hidden layers.
    
    Args:
        input_size: Number of input features
        hidden_layers: List of integers representing neurons in each hidden layer
        output_size: Number of output neurons (default=1 for binary classification)
    
    Returns:
        matplotlib figure object
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Calculate positions for each layer
    layer_spacing = 2
    max_neurons = max([input_size] + hidden_layers + [output_size])
    
    # Generate positions for input layer
    input_pos = {
        f'i{i}': (0, (i - input_size/2) * max_neurons/input_size) 
        for i in range(input_size)
    }
    
    # Generate positions for hidden layers
    hidden_pos = {}
    for l, layer_size in enumerate(hidden_layers):
        for i in range(layer_size):
            hidden_pos[f'h{l}_{i}'] = (
                (l + 1) * layer_spacing,
                (i - layer_size/2) * max_neurons/layer_size
            )
    
    # Generate positions for output layer
    output_pos = {
        f'o{i}': ((len(hidden_layers) + 1) * layer_spacing,
                  (i - output_size/2) * max_neurons/output_size) 
        for i in range(output_size)
    }
    
    # Combine all positions
    pos = {**input_pos, **hidden_pos, **output_pos}
    
    # Add nodes for each layer
    for i in range(input_size):
        G.add_node(f'i{i}', pos=input_pos[f'i{i}'], layer='input')
    
    for l, layer_size in enumerate(hidden_layers):
        for i in range(layer_size):
            G.add_node(f'h{l}_{i}', pos=hidden_pos[f'h{l}_{i}'], layer=f'hidden_{l}')
    
    for i in range(output_size):
        G.add_node(f'o{i}', pos=output_pos[f'o{i}'], layer='output')
    
    # Add edges between layers
    # Input to first hidden layer
    for i in range(input_size):
        for h in range(hidden_layers[0]):
            G.add_edge(f'i{i}', f'h0_{h}')
    
    # Between hidden layers
    for l in range(len(hidden_layers) - 1):
        for h1 in range(hidden_layers[l]):
            for h2 in range(hidden_layers[l + 1]):
                G.add_edge(f'h{l}_{h1}', f'h{l+1}_{h2}')
    
    # Last hidden layer to output
    for h in range(hidden_layers[-1]):
        for o in range(output_size):
            G.add_edge(f'h{len(hidden_layers)-1}_{h}', f'o{o}')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Draw the network
    pos = nx.get_node_attributes(G, 'pos')
    
    # Define colors for each layer
    layer_colors = {
        'input': '#6baed6',     # Blue for input
        'output': '#74c476'     # Green for output
    }
    # Generate colors for hidden layers (orange gradient)
    n_hidden = len(hidden_layers)
    for i in range(n_hidden):
        alpha = (i + 1) / (n_hidden + 1)
        layer_colors[f'hidden_{i}'] = f'#{int(253 * (1-alpha) + 127 * alpha):02x}' + \
                                     f'{int(141 * (1-alpha) + 69 * alpha):02x}' + \
                                     f'{int(60 * (1-alpha) + 0 * alpha):02x}'
    
    # Draw nodes for each layer with different colors
    for layer, color in layer_colors.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == layer]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color,
                             node_size=500, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2)
    
    # Add labels
    labels = {}
    # Input layer labels
    for i in range(input_size):
        if i < 3 or i == input_size - 1:  # Show first 3 and last feature
            labels[f'i{i}'] = f'X{i}'
        elif i == 3:
            labels[f'i{i}'] = '...'
        else:
            labels[f'i{i}'] = ''
    
    # Hidden layer labels
    for l, layer_size in enumerate(hidden_layers):
        for i in range(layer_size):
            labels[f'h{l}_{i}'] = f'H{i}'
    
    # Output layer labels
    for i in range(output_size):
        labels[f'o{i}'] = 'Y'
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Add title and layer labels
    plt.title('Neural Network Architecture', size=16, pad=20)
    
    # Add layer descriptions
    x_positions = [0] + [(i+1)*layer_spacing for i in range(len(hidden_layers))] + \
                 [(len(hidden_layers)+1)*layer_spacing]
    descriptions = ['Input Layer'] + [f'Hidden Layer {i+1}\n({size} neurons)' 
                                    for i, size in enumerate(hidden_layers)] + ['Output Layer']
    
    for x, desc in zip(x_positions, descriptions):
        plt.text(x, -max_neurons/1.5, desc, ha='center', va='center')
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()