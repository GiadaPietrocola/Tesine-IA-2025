"""
Maintenance scenario implementation.
Goal: Visit multiple stations in a specific order.
"""

from typing import Optional, List
from src.models.graph import EnergyGrid
from src.algorithm.astar import AStarPathfinder
from src.algorithm.heuristics import MaintenanceHeuristic


def solve_maintenance_scenario(grid: EnergyGrid, start_station: int,
                             stations_to_visit: List[int]) -> Optional[List[int]]:
    """
    Find optimal path visiting multiple stations in order.
    
    Args:
        grid: The energy grid
        start_station: Starting station ID
        stations_to_visit: List of stations to visit in order
        
    Returns:
        Optional[List[int]]: Complete maintenance route, if found
    """

    # Istanza dell'euristica
    heuristic = MaintenanceHeuristic(grid, stations_to_visit)

    # Inizializzazione dell' A* pathfinder
    pathfinder = AStarPathfinder(grid, heuristic)

    # Lista per memorizzare il percorso completo
    complete_path = []
    current = start_station

    # Per ogni stazione da visitare calcola il percorso
    for next_station in stations_to_visit:
        subpath = pathfinder.find_path(current, next_station)
        if not subpath:
            return None
        complete_path.extend(subpath[:-1])  # Aggiunge i nodi del percorso esludendo l'ultimo per non duplicarlo
                                            # perchè sarà l'inizio del prossimo percordo
        current = next_station
    complete_path.append(stations_to_visit[-1])    # Aggiunge l'ultima stazione da visitare al percorso completo
    return complete_path

