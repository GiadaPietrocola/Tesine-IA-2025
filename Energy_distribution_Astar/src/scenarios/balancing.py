"""
Energy balancing scenario implementation.
Goal: Connect low energy stations efficiently.
"""

from typing import Optional, List
from src.models.graph import EnergyGrid
from src.algorithm.astar import AStarPathfinder
from src.algorithm.heuristics import BalancingHeuristic
import math

def solve_balancing_scenario(grid: EnergyGrid, start_station: int,
                           low_energy_stations: List[int]) -> Optional[List[int]]:
    """
    Find optimal path connecting low energy stations.
    
    Args:
        grid: The energy grid
        start_station: Starting station ID
        low_energy_stations: List of stations needing energy
        
    Returns:
        Optional[List[int]]: Path connecting all required stations, if found
    """
    # TODO: Student Implementation
    # 1. Create BalancingHeuristic instance
    # heuristic = BalancingHeuristic(grid, low_energy_stations)
    
    # 2. Initialize A* pathfinder
    # pathfinder = AStarPathfinder(grid, heuristic)
    
    # 3. Consider:
    #    - Find optimal order to visit stations
    #    - Minimize total path distance
    #    - Consider energy levels when planning route
    #    - Implement nearest neighbor or similar approach
    
    # 4. Example approach:
    # ordered_stations = find_optimal_station_order(grid, start_station, low_energy_stations)
    # return find_path_through_stations(pathfinder, ordered_stations)

    # Uguale a maintenance
    heuristic = BalancingHeuristic(grid, low_energy_stations)

    ordered_stations = find_optimal_station_order(grid, start_station, low_energy_stations)
    pathfinder = AStarPathfinder(grid, heuristic)

    complete_path = []
    current = start_station
    for next_station in ordered_stations:
        subpath = pathfinder.find_path(current, next_station)
        if not subpath:
            return None
        complete_path.extend(subpath[:-1])
        current = next_station
    complete_path.append(ordered_stations[-1])
    return complete_path



def find_optimal_station_order(grid: EnergyGrid, start_station: int, low_energy_stations: List[int]) -> List[int]:
    """
    Find the optimal order of stations to visit based on distance and energy level.

    Args:
        grid: The energy grid
        start_station: Starting station ID
        low_energy_stations: List of stations needing energy

    Returns:
        List[int]: Optimal order of stations to visit
    """


    current_station = start_station
    stations_to_visit = low_energy_stations[:]
    order = []

    # Continua a cercare finché ci sono stazioni da visitare
    while stations_to_visit:
        # Variabili per tenere traccia della stazione più vicina e della distanza minima
        nearest_station = None
        min_distance = float('inf')

        for station in stations_to_visit:
            # Posizioni della stazione corrente e della stazione da visitare
            pos_current = grid.station_data[current_station]['pos']
            pos_station = grid.station_data[station]['pos']

            # Distanza euclidea tra le due stazioni
            distance = math.sqrt((pos_current[0] - pos_station[0]) ** 2 + (pos_current[1] - pos_station[1]) ** 2)

            # Penalità per stazioni con livello energetico più alto
            energy_level = grid.station_data[station]["energy_level"]
            total_cost = distance * energy_level/10

            # Se la stazione è più vicina di quella trovata finora aggiorna
            if total_cost < min_distance:
                min_distance = total_cost
                nearest_station = station


        # Aggiunge la stazione trovata all'ordine e la rimuove dalla lista delle stazioni da visitare
        order.append(nearest_station)
        stations_to_visit.remove(nearest_station)
        # Aggiorna la stazione corrente a quella appena visitata
        current_station = nearest_station

    return order