"""
Heuristic functions for A* pathfinding in energy grid optimization.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from src.models.graph import EnergyGrid
import math
import networkx as nx

class BaseHeuristic(ABC):
    """Base class for heuristic implementations."""
    
    def __init__(self, grid: EnergyGrid):
        """
        Initialize heuristic with grid.
        
        Args:
            grid: The energy distribution grid
        """
        self.grid = grid
    
    @abstractmethod
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost from current to goal node.
        
        Args:
            current: Current station ID
            goal: Goal station ID
            
        Returns:
            float: Estimated cost to goal
        """
        pass


class EmergencyHeuristic(BaseHeuristic):
    """Heuristic for emergency scenario."""
    
    def __init__(self, grid: EnergyGrid, critical_station: int):
        """Initialize emergency heuristic."""
        super().__init__(grid)
        self.critical_station = critical_station
        
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost considering emergency requirements.

        """
        # Posizioni della stazione corrente e della stazione obiettivo
        pos_current = self.grid.station_data[current]['pos']
        pos_goal = self.grid.station_data[goal]['pos']

        # Distanza euclidea
        distance = math.sqrt((pos_current[0] - pos_goal[0]) ** 2 + (pos_current[1] - pos_goal[1]) ** 2)

        # Livello energetico della stazione corrente
        energy_level = self.grid.station_data[current]['energy_level']

        # Fattore di penality direttamente proporzionale al livello di energia
        penality_factor = energy_level/100

        # Fattore aggiuntivo se la stazione è critica
        if self.grid.station_data[current]['is_critical']:
            critical_factor = 0.8
        else: critical_factor = 1

        distance *= penality_factor*critical_factor

        return distance

class MaintenanceHeuristic(BaseHeuristic):
    """Heuristic for maintenance scenario."""
    
    def __init__(self, grid: EnergyGrid, stations_to_visit: List[int]):
        """Initialize maintenance heuristic."""
        super().__init__(grid)
        self.stations_to_visit = stations_to_visit
        
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost considering maintenance requirements.

        """
        # Posizioni della stazione corrente e della stazione obiettivo
        pos_current = self.grid.station_data[current]['pos']
        pos_goal = self.grid.station_data[goal]['pos']

        # Distanza euclidea
        distance = math.sqrt((pos_current[0] - pos_goal[0]) ** 2 + (pos_current[1] - pos_goal[1]) ** 2)

        # Livello energetico della stazione corrente
        energy_level = self.grid.station_data[current]['energy_level']

        # Fattore di penality direttamente proporzionale al livello di energia
        penality_factor = energy_level/50 # 50 non passa per stazione critica, 20 sì

        # Fattore aggiuntivo se la stazione è critica
        if self.grid.station_data[current]['is_critical']:
            critical_factor = 0.5
        else: critical_factor = 1

        distance *= penality_factor*critical_factor

        return distance

class BalancingHeuristic(BaseHeuristic):
    """Heuristic for energy balancing scenario with MST-based estimation."""

    def __init__(self, grid: EnergyGrid, low_energy_stations: List[int]):
        """Initialize balancing heuristic with MST support."""
        super().__init__(grid)
        self.low_energy_stations = low_energy_stations

    def compute_mst_cost(self, current: int, goal: int, remaining_stations: List[int]) -> float:
        """
        Compute the MST cost over the remaining low-energy stations, including the current and goal.
        """
        if not remaining_stations:
            return 0  # Nessun MST necessario se non ci sono stazioni rimanenti

        # Consideriamo solo le stazioni rimanenti, più quella corrente e l'obiettivo
        nodes = remaining_stations + [current, goal]

        # Creiamo un sottografo delle sole stazioni rilevanti
        subgraph = nx.Graph()
        for i in nodes:
            for j in nodes:
                if i != j:
                    pos_i = self.grid.station_data[i]['pos']
                    pos_j = self.grid.station_data[j]['pos']
                    dist = math.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2)
                    subgraph.add_edge(i, j, weight=dist)

        # Calcoliamo l'MST con Prim
        mst = nx.minimum_spanning_tree(subgraph)
        mst_cost = sum(weight for _, _, weight in mst.edges(data="weight"))

        return mst_cost

    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost considering balancing requirements and MST-based path estimation.
        """

        # Livello energetico della stazione corrente
        energy_level = self.grid.station_data[current]['energy_level']

        # Fattore di penality direttamente proporzionale al livello di energia
        penality_factor = energy_level/10

        # Fattore aggiuntivo se la stazione è critica
        if self.grid.station_data[current]['is_critical']:
            critical_factor = 0.4
        else: critical_factor = 1

        # Stazioni a basso livello energetico ancora non visitate
        remaining_stations = [s for s in self.low_energy_stations if s != current]

        # Calcolo dell'MST sulle stazioni rimanenti per stimare il costo restante
        mst_cost = self.compute_mst_cost(current, goal, remaining_stations)

        # Euristica finale: somma della distanza ponderata e dell'MST
        return mst_cost*penality_factor*critical_factor