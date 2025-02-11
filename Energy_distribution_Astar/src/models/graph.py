import networkx as nx
from typing import Dict, List, Tuple, Optional
import random


class EnergyGrid:
    """Represents the energy distribution network."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.station_data = {}
        
    def add_station(self, station_id: int, 
                   position: Tuple[float, float],
                   energy_level: float = 100.0,
                   is_critical: bool = False,
                   status: str = "normal") -> None:
        """
        Add a station to the grid.
        
        Args:
            station_id: Unique identifier for the station
            position: (x, y) coordinates for visualization
            energy_level: Current energy level (0-100)
            is_critical: Whether the station is critical
            status: Station status ("normal", "critical", "low_energy")
        """
        self.graph.add_node(station_id)
        self.station_data[station_id] = {
            'pos': position,
            'energy_level': energy_level,
            'is_critical': is_critical,
            'status': status
        }
        
    def add_connection(self, station1: int, station2: int, 
                      weight: float) -> None:
        """
        Add a connection between stations with associated weight.
        
        Args:
            station1: ID of first station
            station2: ID of second station
            weight: Cost/distance of the connection
        """
        self.graph.add_edge(station1, station2, weight=weight)
        
    @classmethod
    def create_fixed_grid(cls) -> 'EnergyGrid':
        """
        Create a fixed grid with predefined stations and connections.
        
        Returns:
            EnergyGrid: A new grid with fixed layout
        """
        grid = cls()
        
        # Define fixed positions in a clear layout (roughly forming a city grid)
        positions = {
            # Central area (0-3)
            0: (300, 300),  # Central Station
            1: (200, 300),
            2: (400, 300),
            3: (300, 200),
            # North area (4-7)
            4: (200, 500),  # North Station
            5: (300, 500),
            6: (400, 500),
            7: (300, 400),
            # South area (8-11)
            8: (200, 100),
            9: (300, 100),  # South Station
            10: (400, 100),
            11: (300, 0),
            # East area (12-15)
            12: (500, 200),
            13: (500, 300),
            14: (500, 400),
            15: (600, 300),
            # West area (16-19)
            16: (100, 200),
            17: (100, 300),
            18: (100, 400),
            19: (0, 300),
        }
        
        # Define station statuses
        statuses = {
            # Critical stations
            2: {"status": "critical", "energy_level": 30, "is_critical": True},
            15: {"status": "critical", "energy_level": 20, "is_critical": True},
            
            # Low energy stations
            7: {"status": "low_energy", "energy_level": 40, "is_critical": False},
            11: {"status": "low_energy", "energy_level": 35, "is_critical": False},
            16: {"status": "low_energy", "energy_level": 30, "is_critical": False},
            19: {"status": "low_energy", "energy_level": 25, "is_critical": False},
            
            # Normal stations (will be applied to all others)
        }
        
        # Add all stations
        for station_id, pos in positions.items():
            status_data = statuses.get(station_id, {
                "status": "normal",
                "energy_level": 85,
                "is_critical": False
            })
            
            grid.add_station(
                station_id=station_id,
                position=pos,
                energy_level=status_data["energy_level"],
                is_critical=status_data["is_critical"],
                status=status_data["status"]
            )
        
        # Add connections with realistic distances
        connections = [
            # Central connections
            (0, 1, 100), (0, 2, 100), (0, 3, 100), (0, 7, 100),
            # North area
            (4, 5, 100), (5, 6, 100), (7, 5, 100),
            # South area
            (8, 9, 100), (9, 10, 100), (9, 11, 100),
            # East area
            (12, 13, 100), (13, 14, 100), (13, 15, 100),
            # West area
            (16, 17, 100), (17, 18, 100), (17, 19, 100),
            # Cross connections
            (1, 17, 100), (2, 13, 100), (3, 9, 100), (7, 14, 100),
            (4, 18, 150), (6, 14, 150), (8, 16, 150), (10, 12, 150)
        ]
        
        for start, end, weight in connections:
            grid.add_connection(start, end, weight)
            
        return grid
        
    def get_node_positions(self) -> Dict:
        """Return positions of all nodes for visualization."""
        return {node: self.station_data[node]['pos'] 
                for node in self.graph.nodes()}
        
    def get_node_colors(self) -> List:
        """Return colors for nodes based on their status."""
        color_map = {
            "normal": "lightblue",
            "critical": "red",
            "low_energy": "orange"
        }
        return [color_map[self.station_data[node]['status']] 
                for node in self.graph.nodes()]
        
    def get_edge_weights(self) -> Dict:
        """Return all edge weights."""
        return nx.get_edge_attributes(self.graph, 'weight')


    def get_neighbors(self, node: int) -> List[int]:
        """
        Restituisce i vicini di un nodo dato nel grafo.

        Args:
            node: ID del nodo per cui trovare i vicini

        Returns:
            List[int]: Lista dei nodi vicini
        """
        return list(self.graph.neighbors(node))

    def get_distance(self, station1: int, station2: int) -> float:
        """
        Restituisce la distanza (peso) tra due stazioni connesse.

        Args:
            station1: ID della prima stazione
            station2: ID della seconda stazione

        Returns:
            float: Peso dell'arco che collega le due stazioni

        Raises:
            ValueError: Se le stazioni non sono connesse
        """
        if self.graph.has_edge(station1, station2):
            return self.graph[station1][station2]['weight']
        else:
            raise ValueError(f"Le stazioni {station1} e {station2} non sono connesse.")