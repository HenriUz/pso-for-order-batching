from dataclasses import dataclass
from math import exp
from methods.utils import Particle
from process.dataset import Problem
from random import random
from time import perf_counter

@dataclass
class BinaryParticle(Particle):
    selected_aisles: list[int]
    pbest: list[int]
    velocity: list[float]

def generate_particle(problem: Problem) -> BinaryParticle:
    """
    Generates a particle with random aisles, each with a 50% inclusion probability.

    Args:
        problem (Problem): Dataset information.
        
    Returns:
        particle (BinaryParticle): Particle created.
    """

    particle = BinaryParticle(
        aisles_items    = [0] * problem.i,
        number_aisles   = 0,
        objective       = 0.0,
        pbest_obj       = 0.0,
        pbest_n_aisles  = 0,
        selected_aisles = [1 if random() <= 0.5 else 0 for _ in range(problem.a)],
        pbest           = [0] * problem.a,
        velocity        = []
    )
    
    for a in range(problem.a):
        if particle.selected_aisles[a]:
            particle.number_aisles += 1
            particle.pbest[a] = 1

            aisle_items = problem.aisles[a]
            for item in aisle_items:
                particle.aisles_items[item] += aisle_items[item]
    
    particle.objective = problem.objective_function(
        problem.add_orders(particle.aisles_items), particle.number_aisles
    )

    particle.pbest_obj      = particle.objective
    particle.pbest_n_aisles = particle.number_aisles

    return particle

def generate_initial_swarm(
    problem: Problem,
    size: int,
    v_min: float,
    v_max: float,
    swarm: list
) -> tuple[float, int, list[int]]:
    """
    Populates the swarm with randomly generated particles and returns the global best.
    
    The inital velocity of each component of the particle's velocity is generated randomly within the range of the maximum and minumum velocities. Performance is slightly degraded by the addition of another loop to calculate the velocities; this could be avoided by calculating the velocity during particle generation, but to ensure that the particles have the same initial position across all methods (including SBPSO), this step is necessary.

    Args:
        problem (Problem): Dataset information.
        size (int): Swarm size.
        v_min (float): Minimum velocity.
        v_max (float): Maximum velocity.
        swarm (list): Empty list to be populated.
    
    Returns:
        best_particle (tuple[float, int, list[int]]): Objective value, number of aisles, and aisles of the best particle found.
    """
    
    best_position = []
    best_obj      = 0.0
    best_n_aisles = 0
    
    for _ in range(size):
        particle = generate_particle(problem)
        swarm.append(particle)

        if particle.objective > best_obj or (
            particle.objective == best_obj 
            and particle.number_aisles < best_n_aisles
        ):
            best_position = particle.selected_aisles
            best_obj      = particle.objective
            best_n_aisles = particle.number_aisles
    
    for i in range(size):
        swarm[i].velocity = [((v_max - v_min) * random() + v_min) for _ in range(problem.a)]
    
    return (best_obj, best_n_aisles, best_position[:])

def MBPSO(
    problem: Problem, 
    size: int, 
    max_generation: int, 
    w_max: float, 
    w_min: float, 
    c1: float, 
    c2: float, 
    v_min: float, 
    v_max: float, 
    r_mu: float
) -> None:
    """
    Modified Binary PSO for order batching. Operates on the aisle search space; orders are decoded greedily from the selected aisles after each position update.

    Transfer function (V-shaped) [Yang et al., 2014]:
    - If v <= 0: T(v) = 1 - 2 / (1 + exp(-v))
    - If v > 0: T(v) = 2 / (1 + exp(-v)) - 1

    Result is written directly to `problem.result`.

    Args:
        problem (Problem): Dataset information.
        size (int): Swarm size.
        max_generation (int): Maximum number of iterations.
        w_max (float): Initial inertia.
        w_min (float): Final inertia.
        c1 (float): Cognitive acceleration coefficient.
        c2 (float): Social acceleration coefficient.
        v_min (float): Minimum velocity.
        v_max (float): Maximum velocity.
        r_mu (float): Probability of mutation.
    """

    start = perf_counter()
    
    aisles_data = problem.aisles

    swarm = []
    best_obj, best_n_aisles, best_aisles = (
        generate_initial_swarm(problem, size, v_min, v_max, swarm)
    )

    generation = 0
    while generation < max_generation:
        # Linear inertia decay: high exploration early, intensification later.
        w = (w_max - w_min) * ((max_generation - generation) / max_generation) + w_min

        for i in range(size):
            particle = swarm[i]

            selected = particle.selected_aisles
            vel_list = particle.velocity
            pbest    = particle.pbest

            for a in range(problem.a):
                prev_bit = selected[a]

                # Velocity update (standard PSO with clamping).
                vel = w * vel_list[a]
                vel += c1 * random() * (pbest[a]       - prev_bit)
                vel += c2 * random() * (best_aisles[a] - prev_bit)
                
                if vel < v_min:
                    vel = v_min
                elif vel > v_max:
                    vel = v_max
                
                vel_list[a] = vel

                # V-shaped transfer -> stochastic bit update.
                if vel <= 0:
                    transfer = 1.0 - 2.0 / (1.0 + exp(-vel))
                    new_bit  = 0
                else:
                    transfer = 2.0 / (1.0 + exp(-vel)) - 1.0
                    new_bit  = 1

                if random() <= transfer:
                    selected[a] = new_bit

                # Mutation: random bit flip with probability r_mu.
                if random() < r_mu:
                    selected[a] ^= 1

                # Incremental update of available items (avoids full recomputation).
                if selected[a] != prev_bit:
                    aisle_items = aisles_data[a]
                    
                    if prev_bit:  # Aisle deselected.
                        particle.number_aisles -= 1
                        for item in aisle_items:
                            particle.aisles_items[item] -= aisle_items[item]
                    else:         # Aisle selected.
                        particle.number_aisles += 1
                        for item in aisle_items:
                            particle.aisles_items[item] += aisle_items[item]

        for i in range(size):
            particle = swarm[i]

            particle.objective = problem.objective_function(
                problem.add_orders(particle.aisles_items), particle.number_aisles
            )

            if particle.objective > particle.pbest_obj or (
                particle.objective == particle.pbest_obj 
                and particle.number_aisles < particle.pbest_n_aisles
            ):
                particle.pbest          = particle.selected_aisles[:]
                particle.pbest_obj      = particle.objective
                particle.pbest_n_aisles = particle.number_aisles
            
            if particle.objective > best_obj or (
                particle.objective == best_obj 
                and particle.number_aisles < best_n_aisles
            ):
                best_aisles   = particle.selected_aisles[:]
                best_obj      = particle.objective
                best_n_aisles = particle.number_aisles
        
        generation += 1

    end = perf_counter()

    # Assembling the available items for the best overall result -> faster than copying them over from generation to generation.
    best_aisles_items = [0] * problem.i
    for a in range(problem.a):
        if best_aisles[a]:
            aisle = aisles_data[a]
            for item in aisle:
                best_aisles_items[item] += aisle[item]

    problem.result["orders"]    = problem.view_orders(best_aisles_items)
    problem.result["aisles"]    = [a for a in range(problem.a) if best_aisles[a]]
    problem.result["objective"] = best_obj
    problem.result["time"]      = end - start

def MBPSOzt(
    problem: Problem,
    size: int,
    max_generation: int,
    w_max: float,
    w_min:float,
    c1: float,
    c2: float,
    v_min: float,
    v_max: float,
    r_mu: float,
    k: float
) -> None:
    """
    Modified Binary PSO with Zhang's transfer function for order batching. Operates on the aisle search space; orders are decoded greedily from the selected aisles after each position update.

    Transfer function (V-shaped) [Zhang et al., 2020]:
    - T(v) = 1 - exp(-`k` * |v|)

    Result is written directly to `problem.result`.

    Args:
        problem (Problem): Dataset information.
        size (int): Swarm size.
        max_generation (int): Maximum number of iterations.
        w_max (float): Initial inertia.
        w_min (float): Final inertia.
        c1 (float): Cognitive acceleration coefficient.
        c2 (float): Social acceleration coefficient.
        v_min (float): Minimum velocity.
        v_max (float): Maximum velocity.
        r_mu (float): Probability of mutation.
        k (float): Transfer function parameter (> 0).
    """

    if k <= 0:
        return

    start = perf_counter()
    
    aisles_data = problem.aisles

    swarm = []
    best_obj, best_n_aisles, best_aisles = (
        generate_initial_swarm(problem, size, v_min, v_max, swarm)
    )

    generation = 0
    while generation < max_generation:
        # Linear inertia decay: high exploration early, intensification later.
        w = (w_max - w_min) * ((max_generation - generation) / max_generation) + w_min
        
        for i in range(size):
            particle = swarm[i]

            selected = particle.selected_aisles
            vel_list = particle.velocity
            pbest    = particle.pbest
            
            for a in range(problem.a):
                prev_bit = selected[a]

                # Velocity update (standard PSO with clamping).
                vel = w * vel_list[a]
                vel += c1 * random() * (pbest[a]       - prev_bit)
                vel += c2 * random() * (best_aisles[a] - prev_bit)
                
                if vel < v_min:
                    vel = v_min
                elif vel > v_max:
                    vel = v_max
                
                vel_list[a] = vel

                # V-shaped transfer -> stochastic bit update.
                transfer_vel = vel
                new_bit = 1
                if vel <= 0:
                    transfer_vel = -vel
                    new_bit = 0

                transfer = 1 - exp(-k * transfer_vel)
                
                if random() <= transfer:
                    selected[a] = new_bit
                    
                # Mutation: random bit flip with probability r_mu.
                if random() < r_mu:
                    selected[a] ^= 1

                # Incremental update of available items (avoids full recomputation).
                if selected[a] != prev_bit:
                    aisle_items = aisles_data[a]
                    
                    if prev_bit:  # Aisle deselected.
                        particle.number_aisles -= 1
                        for item in aisle_items:
                            particle.aisles_items[item] -= aisle_items[item]
                    else:         # Aisle selected.
                        particle.number_aisles += 1
                        for item in aisle_items:
                            particle.aisles_items[item] += aisle_items[item]

        for i in range(size):
            particle = swarm[i]

            particle.objective = problem.objective_function(
                problem.add_orders(particle.aisles_items), particle.number_aisles
            )            

            if particle.objective > particle.pbest_obj or (
                particle.objective == particle.pbest_obj 
                and particle.number_aisles < particle.pbest_n_aisles
            ):
                particle.pbest          = particle.selected_aisles[:]
                particle.pbest_obj      = particle.objective
                particle.pbest_n_aisles = particle.number_aisles

            if particle.objective > best_obj or (
                particle.objective == best_obj 
                and particle.number_aisles < best_n_aisles
            ):
                best_aisles   = particle.selected_aisles[:]
                best_obj      = particle.objective
                best_n_aisles = particle.number_aisles
                
        generation += 1

    end = perf_counter()

    # Assembling the available items for the best overall result -> faster than copying them over from generation to generation.
    best_aisles_items = [0] * problem.i
    for a in range(problem.a):
        if best_aisles[a]:
            aisle = aisles_data[a]
            for item in aisle:
                best_aisles_items[item] += aisle[item]

    problem.result["orders"]    = problem.view_orders(best_aisles_items)
    problem.result["aisles"]    = [a for a in range(problem.a) if best_aisles[a]]
    problem.result["objective"] = best_obj
    problem.result["time"]      = end - start