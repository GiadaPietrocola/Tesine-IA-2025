"""
A* pathfinding algorithm implementation for energy grid optimization.
"""
import heapq
from typing import List, Optional, Dict, Set, Callable

from setuptools.command.alias import format_alias

from src.models.graph import EnergyGrid
from src.algorithm.heuristics import BaseHeuristic

class AStarPathfinder:
    """A* pathfinding implementation for energy grid."""
    
    def __init__(self, grid: EnergyGrid, heuristic: BaseHeuristic):
        """
        Initialize pathfinder with grid and heuristic.
        
        Args:
            grid: The energy distribution grid
            heuristic: Heuristic function implementation
        """
        self.grid = grid
        self.heuristic = heuristic
    
    def find_path(self, start: int, goal: int) -> Optional[List[int]]:
        """
        Find optimal path between start and goal stations.
        
        Args:
            start: Starting station ID
            goal: Goal station ID
            
        Returns:
            Optional[List[int]]: Path from start to goal if found, None otherwise
        """
        
        # 1. Inizializza le strutture dati
        open_set = [(0, start)]                                   # Lista open set con nodo iniziale
        closed_set: Set[int] = set()                              # Insieme dei nodi già esplorati
        g_scores = {start: 0}                                     # Dizionario con il costo effettivo minimo per ogni nodo
        f_scores = {start: self.heuristic.estimate(start, goal)}  # Costo totale stimato (g + h)
        came_from = {start: None}                                 # Dizionario per ricostruire il percorso



        while open_set:
            # Estrae il nodo con il costo stimato (f_score) più baso dalla cosa
            current = heapq.heappop(open_set)[1]

            print(f"Exploring node {current}")
            print(f"Current cost: {g_scores[current]}")
            print(f"Heuristic: {self.heuristic.estimate(current, goal)}")

            # Se il nodo corrente è quello obiettivo, ricostruisce il percorso
            if current == goal:
                print("Goal reached")
                return self._reconstruct_path(came_from, current)

            # Aggiunge il nodo corrente al set di nodi esplorati
            closed_set.add(current)

            # Itera sui nodi vicini al nodo corrente
            for next_node in self.grid.get_neighbors(current):

                #Ignora i nodi già esplorati
                if next_node in closed_set:
                    continue

                # Calcola il costo effettivo per raggiungere il nodo vicino
                tentative_g_score = g_scores[current]+ self.grid.get_distance(current,next_node)

                # Se trova un percorso migliore (con un costo più basso) aggiorna
                if next_node not in g_scores or tentative_g_score < g_scores[next_node]:
                    g_scores[next_node] = tentative_g_score
                    f_scores[next_node] = tentative_g_score + self.heuristic.estimate(next_node, goal)
                    # Inserisce il nodo nella coda di priorià
                    heapq.heappush(open_set, (f_scores[next_node], next_node))
                    # Memorizza il nodo corrente come predecessore
                    came_from[next_node] = current

        # Ritorna None se non esiste un percorso valido
        return None

    
    def _reconstruct_path(self, came_from: Dict[int, int], 
                         current: int) -> List[int]:
        """
        Reconstruct path from came_from dictionary.
        
        Args:
            came_from: Dictionary tracking path predecessors
            current: Current (goal) node
            
        Returns:
            List[int]: Reconstructed path
        """
        path = []  #Lista per memorizzare il percorso

        while current is not None:
            path.append(current) # Aggiunge il nodo corrente al percorso
            current = came_from.get(current) # Passa al predecessore
        return path[::-1]  # Inverte il percorso


