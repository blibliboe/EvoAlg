import numpy as np
import numba as nb
from numba import types as nty

def get_variation_fn(
    population_size: int,
    max_expression_size: int,
    num_constants: int,
    library_size: int,
    p_crossover: float,
    scaling_factor: float,
    linear_scaling: bool,
    evaluate_individual: callable,
    evaluate_population: callable,
    progress_dependent_crossover: bool
):
    @nb.jit((
            nty.Array(nty.float32, 1, "C"),
    ), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=True,
        parallel=False)
    def get_node_depths(tree):
        """
        Returns a list of the depth of each node in the given tree representation.

        Parameters:
        tree (np.ndarray): The tree representation in prefix notation.
        max_expression_size (int): The maximum size of the expression.
        num_operators (int): The number of operators in the operator set.
        arity (np.ndarray): A 1D NumPy array containing the arity of each operator.

        Returns:
        np.ndarray: A 1D NumPy array containing the depth of each node in the tree.
        """

        # OPERATORS = [
        #     # Add a tuple consisting of (op_id, format string, vectorized function)
        #     # for each operator you want to use (arity is derived from the format string)
        #     # Note: currently, only numpy functions are supported!
        #     ("+", "({} + {})", np.add),
        #     ("-", "({} - {})", np.subtract),
        #     ("*", "({} * {})", np.multiply),
        #     ("/", "({} / {})", np.divide),
        #     ("sin", "sin({})", np.sin),
        #     ("cos", "cos({})", np.cos),
        #     ("exp", "exp({})", np.exp),
        #     ("log", "log({})", np.log),
        #     ("sqrt", "sqrt({})", np.sqrt)
        # ]
        OPERATOR_ARITIES = np.zeros(8, dtype=np.int32)
        OPERATOR_ARITIES[0] = 2
        OPERATOR_ARITIES[1] = 2
        OPERATOR_ARITIES[2] = 2
        OPERATOR_ARITIES[3] = 2
        OPERATOR_ARITIES[4] = 1
        OPERATOR_ARITIES[5] = 1
        OPERATOR_ARITIES[6] = 1
        OPERATOR_ARITIES[7] = 1
        OPERATOR_ARITIES[8] = 1

        node_depths = np.zeros(max_expression_size, dtype=np.int32)
        stack = []
        stack.append(0)

        for j in range(max_expression_size):
            if len(stack) == 0:
                break
            node = abs(int(tree[j]))
            if node < 8:
                # If the node is an operator, push its depth and update the depths of its children
                depth = stack.pop()
                node_depths[j] = depth

                i = 0
                while i < OPERATOR_ARITIES[node]:
                    stack.append(depth + 1)
                    i += 1

            else:
                # If the node is a terminal, assign its depth
                depth = stack.pop()
                node_depths[j] = depth
        return node_depths

    @nb.jit((
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C"),
        nty.Array(nty.float32, 2, "C", readonly=True),
        nty.Array(nty.float32, 1, "C", readonly=True),
        nb.typeof(np.random.Generator(np.random.Philox())),
        nty.float32
        ), nopython=True, nogil=True, fastmath={"nsz", "arcp", "contract", "afn"}, error_model="numpy", cache=True, parallel=False)
    def perform_variation(structures, constants, fitness, trial_structures, trial_constants, trial_fitness, X, y, rng, progress):
        """Performs a variation step and returns the number of fitness evaluations performed."""

        all_depths = [get_node_depths(structures[i]) for i in range(population_size)]


        for i in range(population_size):
            r0 = r1 = r2 = i
            while r0 == i:                          r0 = rng.integers(0, population_size)
            while r1 == i or r1 == r0:              r1 = rng.integers(0, population_size)
            while r2 == i or r2 == r0 or r2 == r1:  r2 = rng.integers(0, population_size)
            j_rand: np.int32  = rng.integers(0, max_expression_size + num_constants)

            # The function definitions look a bit weird because numba does not properly support accessing variables
            # outside of the scope of the lambda for some reason.
            if progress_dependent_crossover:
                # Progress-dependent crossover probability
                # High-level nodes (j < max_expression_size // 2) have crossover probability decreasing from 0.3 to 0.03
                # Low-level nodes (j >= max_expression_size // 2) have crossover probability increasing from 0.03 to 0.3
                crossover_probability = lambda i, j, max_expression_size, p_crossover, progress, all_depths: (0.3 - 0.27 * progress) if all_depths[i][j] < np.max(all_depths[i]) // 2 else (0.03 + 0.27 * progress)
            else:
                crossover_probability = lambda i, j, max_expression_size, p_crossover, progress, all_depths: p_crossover

            # construct trial population
            for j in range(structures.shape[1]):
                if rng.random() < crossover_probability(i, j, max_expression_size, p_crossover, progress, all_depths) or j == j_rand:
                    trial_structures[i, j] = structures[r0, j] + scaling_factor * (structures[r1, j] - structures[r2, j])
                    # repair as per Eq 8 (https://doi.org/10.1145/1389095.1389331)
                    v_abs = np.abs(trial_structures[i, j])
                    v_floored_abs = np.floor(v_abs)
                    trial_structures[i, j] = (v_floored_abs % library_size) + (v_abs - v_floored_abs)
                else:
                    trial_structures[i, j] = structures[i, j]
            
            if j_rand > max_expression_size:
                j_rand -= max_expression_size
            
            for j in range(constants.shape[1]):
                if rng.random() < crossover_probability(i, j, max_expression_size, p_crossover, progress, all_depths)  or j == j_rand:
                    trial_constants[i, j] = constants[r0, j] + scaling_factor * (constants[r1, j] - constants[r2, j])
                else:
                    trial_constants[i, j] = constants[i, j]

        evaluate_population(trial_structures, trial_constants, trial_fitness, X, y, linear_scaling)
        return population_size



    return perform_variation
