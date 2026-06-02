from dataclasses import dataclass
from math import floor
from methods.utils import Particle
from process.dataset import Problem
from random import random, sample
from time import perf_counter

@dataclass
class SetParticle(Particle):
    x    : set[int]
    pbest: set[int]

def generate_particle(problem: Problem) -> SetParticle:
    """
    Generates a particle with random aisles, each with a 50% inclusion probability.

    Args:
        problem (Problem): Dataset information.
    
    Returns:
        particle (SetParticle): Particle created.
    """

    particle = SetParticle(
        aisles_items  = [0] * problem.i,
        number_aisles = 0,
        objective     = 0.0,
        pbest_obj     = 0.0,
        x             = {a for a in range(problem.a) if random() <= 0.5},
        pbest         = set()
    )

    particle.pbest = particle.x.copy()
    for aisle in particle.x:
        particle.number_aisles += 1

        aisle_items = problem.aisles[aisle]
        for item in aisle_items:
            particle.aisles_items[item] += aisle_items[item]

    particle.objective = problem.objective_function(
        problem.add_orders(particle.aisles_items), 
        particle.number_aisles
    )
    particle.pbest_obj = particle.objective
    
    return particle

def generate_initial_swarm(
    problem: Problem,
    size: int,
    swarm: list
) -> tuple[float, set[int]]:
    """
    Populates the swarm with randomly generated particles and returns the global best.

    Args:
        problem (Problem): Dataset information.
        size (int): Swarm size.
        swarm (list): Empty list to be populated.

    Returns:
        best_particle (tuple[float, set[int]]): Objective value and aisle set of the best particle found.
    """

    gbest     = set()
    gbest_obj = 0.0

    for _ in range(size):
        particle = generate_particle(problem)
        swarm.append(particle)

        if particle.objective >= gbest_obj:
            gbest     = particle.x
            gbest_obj = particle.objective

    return gbest_obj, gbest.copy()

def scalar_multiplication(
    scalar: float,
    velocity: set[tuple[str, int]]
) -> set[tuple[str, int]]:
    """
    Multiplies a velocity by a scalar value. The result is a new velocity with `floor(scalar * |velocity|)` random elements of `velocity`.

    The scalar must be in [0, 1]: 1 returns the full velocity, and 0 returns an empty set.

    Args:
        scalar (float): Scaling factor in [0, 1].
        velocity (set[tuple[str, int]]): Set of (operator, aisle) operations.
    
    Returns:
        subset (set[tuple[str, int]]): Randomly sampled subset of the velocity.
    """

    if scalar < 0 or scalar > 1:
        return set()
    
    k = floor(scalar * len(velocity))
    return set(sample(sorted(velocity), k=k))

def difference_in_positions(
    target: set[int],
    current: set[int]
) -> set[tuple[str, int]]:
    """
    Computes the velocity needed to transform `current` into `target`.

    Elements only in `target` become additions; elements only in `current` become removals.

    Args:
        target (set[int]): Desired position.
        current (set[int]): Position to be transformed.

    Returns:
        velocity (set[tuple[str, int]]): Set of (operator, aisle) operations. 
    """
    
    additions = {("+", aisle) for aisle in target  - current}
    removals  = {("-", aisle) for aisle in current - target}
    return additions | removals

def number_of_elements(
    beta: float,
    reference_set: set[int]
) -> int:
    """
    Stochastically determines how many elements to operate on, bounded by the set size. Uses `floor(beta)` with a probabilistic +1.
    
    Args:
        beta (float): Expected number of elements.
        reference_set (set[int]): Set whose size serves as the upper bound.
    
    Returns:
        n_elements (int): Number of elements to operate on.
    """
    
    count = floor(beta)
    if random() < beta - count:
        count += 1
    
    length = len(reference_set)
    if count < length:
        return count
    return length

def k_tournament_selection(
    problem: Problem,
    particle: SetParticle,
    candidates: set[int],
    n_to_add: int,
    k: int
) -> set[tuple[str, int]]:
    """
    Greedily selects `n_to_add` aisles from `candidates` using tournament selection.

    For each slot, `k` candidates are sampled and the one that best improves the objective is chosen.

    Args:
        problem (Problem): Dataset information.
        particle (SetParticle): Current particle.
        candidates (set[int]): Aisles not present in `x union pbest union gbest`.
        n_to_add (int): Number of aisles to select.
        k (int): Tournament size.
    
    Returns:
        additions (set[tuple[str, int]]): Addition operations for the selected aisles.
    """
    
    additions = set()
    remaining = sorted(candidates)
    
    # Incremental item availability: updated as aisles are committed each round.
    running_items = particle.aisles_items.copy()

    # n_aisles increases by 1 each round; precompute the value for this round.
    n_aisles = particle.number_aisles + 1

    for _ in range(n_to_add):
        length = len(remaining)
        if k < length:
            tournament_size = k
        else:
            tournament_size = length
        
        contestants = sample(remaining, tournament_size)
        
        best_aisle = -1
        best_obj   = -1
        
        for aisle in contestants:
            temp_items = running_items.copy()
            for item in problem.aisles[aisle]:
                temp_items[item] += problem.aisles[aisle][item]
            
            obj = problem.objective_function(problem.add_orders(temp_items), n_aisles)

            if obj > best_obj:
                best_aisle = aisle
                best_obj   = obj

        additions.add(("+", best_aisle))
        remaining.remove(best_aisle)
        
        # Commit the selected aisle so the next round evaluates on top of it.
        aisle_items = problem.aisles[best_aisle]
        for item in aisle_items:
            running_items[item] += aisle_items[item]

        n_aisles += 1

    return additions

def removal_of_elements(
    consensus_set: set[int],
    n_to_remove: int
) -> set[tuple[str, int]]:
    """
    Randomly selects `n_to_remove` aisles from the consensus intersection to remove.
    
    Args:
        consensus_set (set[int]): Aisles present in `x intersect pbest intersect gbest`.
        n_ro_remove (int): Number of aisles to remove.
    
    Returns:
        removals (set[tuple[str, int]]): Removal operations for the selected aisles.
    """
    
    selected = sample(sorted(consensus_set), k=n_to_remove)
    return {("-", aisle) for aisle in selected}

def SBPSO(
    problem: Problem,
    size: int,
    generations: int,
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    k: int
) -> None:
    """
    Set-Based PSO for the order batching. Operates on the aisle search space; orders are decoded greedily from the selected aisles after each position update.

    `c1` and `c2` must be in [0, 1] (scalar multipliers for set velocities). `c3` and `c4` control the expected number of random additions and removals; they are implicitly bounded by the relevant set sizes inside the called functions.

    Result is written directly to `problem.result`.
    
    Args:
        problem (Problem): Dataset information.
        size (int): Swarm size.
        generations (int): Maximum number of iterations.
        c1 (float): Cognitive acceleration coefficient (in [0, 1]).
        c2 (float): Social acceleration coefficient (in [0, 1]).
        c3 (float): Random addition coefficient.
        c4 (float): Random removal coefficient.
        k (int): Tournament size for k-tournament selection.
    """
    
    if not (0 <= c1 <= 1) or not (0 <= c2 <= 1):
        return
    
    start = perf_counter()

    aisles_data = problem.aisles

    swarm    = []
    universe = set(range(problem.a))

    gbest_obj, gbest = (
        generate_initial_swarm(problem, size, swarm)
    )

    for _ in range(generations):
        for i in range(size):
            particle = swarm[i]

            # Cognitive component: pull toward personal best.
            cognitive_velocity = scalar_multiplication(
                c1 * random(),
                difference_in_positions(particle.pbest, particle.x)
            )

            # Social component: pull toward global best.
            social_velocity = scalar_multiplication(
                c2 * random(),
                difference_in_positions(gbest, particle.x)
            )

            # Exploration: add aisles absent from all three reference sets.
            external_aisles  = universe - (particle.x | particle.pbest | gbest)
            random_additions = k_tournament_selection(
                problem, particle, external_aisles,
                number_of_elements(c3 * random(), external_aisles),
                k
            )

            # Diversity: remove aisles present in all three reference sets.
            consensus_aisles = particle.x & particle.pbest & gbest
            random_removals  = removal_of_elements(
                consensus_aisles,
                number_of_elements(c4 * random(), consensus_aisles)
            )

            velocity = cognitive_velocity | social_velocity | random_additions | random_removals

            for op, aisle in velocity:
                aisle_items = aisles_data[aisle]
                if op == "+":
                    particle.number_aisles += 1
                    particle.x.add(aisle)

                    for item in aisle_items:
                        particle.aisles_items[item] += aisle_items[item]
                else:
                    particle.number_aisles -= 1
                    particle.x.remove(aisle)

                    for item in aisle_items:
                        particle.aisles_items[item] -= aisle_items[item]

        for i in range(size):
            particle = swarm[i]

            particle.objective = problem.objective_function(
                problem.add_orders(particle.aisles_items), particle.number_aisles
            )

            if particle.objective >= particle.pbest_obj:
                particle.pbest     = particle.x.copy()
                particle.pbest_obj = particle.objective

            if particle.objective >= gbest_obj:
                gbest     = particle.x.copy()
                gbest_obj = particle.objective

    end = perf_counter()

    # Assembling the available items for the best overall result. Faster than copying them over from generation to generation.
    gbest_aisles_items = [0] * problem.i
    for a in range(problem.a):
        if a in gbest:
            aisle = aisles_data[a]
            for item in aisle:
                gbest_aisles_items[item] += aisle[item]

    problem.result["orders"]    = problem.view_orders(gbest_aisles_items)
    problem.result["aisles"]    = list(gbest)
    problem.result["objective"] = gbest_obj
    problem.result["time"]      = end - start