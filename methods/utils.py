from dataclasses import dataclass

@dataclass
class Particle():
    """
    Represents a particle in the swarm. It contains attributes useful for both PSO paradigms.

    Attributes:
        aisles_items (list[int]): Number of items available in selected aisles.
        number_aisles (int): Number of aisles selected.
        objective (float): Value of the objective function for the particle.
        pbest_obj (float): Value of the objective function for the best historical position.
        pbest_n_aisles (int): Number of aisles in the best historical position.
    """
    
    aisles_items: list[int]
    number_aisles: int
    objective: float
    pbest_obj: float
    pbest_n_aisles: int