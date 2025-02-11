"""
Emergency scenario implementation.
Goal: Reach the most critical station in minimum time.
"""

from typing import Optional, List
from src.models.graph import EnergyGrid
from src.algorithm.astar import AStarPathfinder
from src.algorithm.heuristics import EmergencyHeuristic


def solve_emergency_scenario(grid: EnergyGrid, start_station: int, 
                           critical_station: int) -> Optional[List[int]]:
    """
    Find the optimal path from start to critical station.
    
    Args:
        grid: The energy grid
        start_station: Starting station ID
        critical_station: Critical station ID to reach
        
    Returns:
        Optional[List[int]]: Path from start to critical station, if found
    """
    # TODO: Student Implementation
    # 1. Create EmergencyHeuristic instance
    # heuristic = EmergencyHeuristic(grid, critical_station)
    
    # 2. Initialize A* pathfinder with the heuristic
    # pathfinder = AStarPathfinder(grid, heuristic)
    
    # 3. Consider:
    #    - Minimize time to reach critical station
    #    - Account for energy consumption
    #    - Handle path constraints
    
    # 4. Find and return the optimal path
    # return pathfinder.find_path(start_station, critical_station)

    # Crea un' istanza dell'euristica
    heuristic = EmergencyHeuristic(grid, critical_station)

    # Inizializza l'A* pathfinder con l'eurisica
    pathfinder = AStarPathfinder(grid, heuristic)

    # Trova il percorso ottimale
    path = pathfinder.find_path(start_station, critical_station)

    return path


