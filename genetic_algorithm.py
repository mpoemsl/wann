""" Contains functions to implement genetic algorithm for WANNs. """

from individuum import Individuum

from copy import deepcopy

import numpy as np


N_CROSS_POINTS = 4


def evolve_population(population, eval_scores, cull_ratio=0.1, elite_ratio=0.1, tournament_size=5, gen=-1, **hyper):
    """ Evolves a complete population. 
    Eliminates weak architectures and allows strong architectures to breed.

    Returns:
    A new population (np.array) of evolved kids - the new generation.

    Parameters:
    population  -   (np.array) the population to evolve
    eval_scores -   (np.array) fitness of the individuals (the higher, the better) 
    cull_ratio  -   [0,1] percentage of population who are not allowed to breed
    elite_ratio -   [0,1] percentage of population who pass on to the new population immediately and unchanged.
                    (elite will do breeding, too)
    tournament_size - (int) number of individuals competing to become the parent of a kid
    gen - (int) number of current generation (-1 if not given)
    """

    # throw error if population and eval_scores do not have the same length
    if len(population) != len(eval_scores):
        raise Exception("population and eval_scores must have the same length")

    new_population = []
    
    ranked_population = population[np.argsort(eval_scores)] # ascending order, worst first, best last 

    # save best individuum of this generation if generation number is given
    if gen > 0:
        ranked_population[-1].save_to("best_individuums/{}/best_gen_{}".format(hyper["experiment_name"], gen))

    # let the fittest x % pass unchanged
    elite_tresh = int(elite_ratio * len(population)) # how many indviduals are passed on
    fittest_indivs = population[-elite_tresh:]
    new_population.extend([deepcopy(indiv) for indiv in fittest_indivs])
    
    # eliminate the unfittest x % 
    cull_tresh = int(cull_ratio * len(population)) # how many individuals are eliminated
    ranked_breeding_pool = ranked_population[cull_tresh:]

    # number of kids that must be generated such that len(new_population) == len(population)
    num_kids = len(population) - elite_tresh

    # tournament selection of parents for each kid
    # pools for the parent A (or B) group with randomly generated indices 
    parent_a_pool = np.random.randint(len(ranked_breeding_pool), size=(num_kids, tournament_size))
    parent_b_pool = np.random.randint(len(ranked_breeding_pool), size=(num_kids, tournament_size))
    
    # choose best parent/architecture from each pool and let them breed
    for pool_a, pool_b in zip(parent_a_pool, parent_b_pool):
        # the best individual is always that with the highest index, because
        # the higher the index, the higher the ranking in ranked_population
        worse_parent_ix, better_parent_ix = sorted([np.max(pool_a), np.max(pool_b)])

        worse_parent, better_parent = ranked_breeding_pool[[worse_parent_ix, better_parent_ix]]

        kid = breed(worse_parent, better_parent, **hyper)

        new_population.append(kid)

    return np.array(new_population)


def breed(worse, better, prob_crossover=0.8, **hyper):
    """ Let two parents breed. Activation functions and nodes are taken by the better parent.
    Only the connections are a mix of both parents. Better parent can also breed alone, see parameter autogamy.

    Returns:
    One kid of the class Individuum.
    
    Parameters:
    worse       -   (Individuum) the worse parent with a lower (or equal) fitness rank than "better"
    better      -   (Individuum) the better parent with a higher (or equal) fitness rank than "worse"
    prob_crossover -   [0,1] probability that the both parents do a crossover and breed (and not one alone).
                       Otherwise, the new kid is a mutated version of the better parent.
    num_cross_points - (int) number of crosspoints for the crossover of the two parents
    """

    # to certain percentage let better genome pass without breeding (later: mutation!)
    if np.random.random() > prob_crossover:
        kid = deepcopy(better)
    else:  # crossover of both parents

        # adapt the structure of the worse to the better parent
        # makes the crossover possible
        worse.adapt_structure_to(better)

        # get genomes
        better_gen = better.get_genome()
        worse_gen = worse.get_genome()
        
        # cross genomes
        kids_gens = crossover(better_gen, worse_gen) # returns two child genomes

        # choose one of the two genomes
        chosen_genome = kids_gens[np.random.random() > 0.5]

        # make new individuum with same structure and activations as better
        kid = Individuum(**hyper)
        kid.adapt_structure_to(better)
        kid.set_activations(better.get_activations())

        # give kid connections from mixed genome
        kid.set_genome(chosen_genome)

    mutate(kid, **hyper)

    return kid


def mutate(individuum, prob_add_node=0.3, prob_add_con=0.4, prob_change_activation=0.3, **kwargs):
    """ Mutates an Individuum. Performs exactly one mutation, which can be either 
    adding a node or adding a connection or changing the activation function.
    
    Parameters:
    individuum      -   (Individuum) that should get a mutation
    prob_add_nod    -   [0,1] probability that a node is added
    prob_add_con    -   [0,1] probability that a connection is added
    prob_change_activation - [0,1] probability that an acitvation function is changed    
    """
    # choose mutation type
    probs = np.array([prob_add_node, prob_add_con, prob_change_activation])
    mutation_type = np.random.choice(["add_node", "add_conn", "change_activation"], p=probs/probs.sum())
    
    # perform mutation
    if mutation_type == "add_node":
      individuum.add_connection()
    elif mutation_type == "add_conn":
      individuum.add_node()
    elif mutation_type == "change_activation":
      individuum.change_activation()
    else:
      raise Exception("Invalid mutation type!")


def crossover(genome1, genome2):
    """ Performs a crossover of two genomes with a certain number of cross points.
    
    Returns:
    The two resulting new genomes (np.array)
    
    Parameters:
    genome1         -   (np.array) consisting of 0s and 1s. One of the two genomes taking part in the crossover
    genome2         -   (np.array) consisting of 0s and 1s. One of the two genomes taking part in the crossover
    """
    # throw error if genome1 and genome2 do not have the same length
    if len(genome1) != len(genome2):
        raise Exception("Genomes must have the same length!")

    # get the indices where the crossover points are
    cross_points = np.array([int(len(genome1) * (i / (N_CROSS_POINTS + 1))) for i in range(1, N_CROSS_POINTS + 1)])

    # shift cross points randomly
    cross_points += np.random.randint(-cross_points[0] + 1, cross_points[0])
  
    # swap the genomes after each crossover point
    kid1 = np.copy(genome1)
    kid2 = np.copy(genome2)

    swap_flag = True

    for cross_point in cross_points:

        if swap_flag:
            kid1[cross_point:] = genome2[cross_point:]
            kid2[cross_point:] = genome1[cross_point:]
        else:
            kid1[cross_point:] = genome1[cross_point:]
            kid2[cross_point:] = genome2[cross_point:]

        swap_flag = not swap_flag

    return kid1, kid2

