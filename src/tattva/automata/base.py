# ---INFO-----------------------------------------------------------------------

# --Authors        |Last Updated |tattva-automata-base
# --Aditya Prakash |10-05-2023   |stable

# --Assumptions
# --Expecting channel-last grid_shape in BaseAutomaton
# ---DEPENDENCIES---------------------------------------------------------------
import jax.numpy as jnp


# ---BASE AUTOMATA--------------------------------------------------------------
class BaseAutomaton:
    def __init__(self):
        self.grid = None

    def init_grid(self, grid):
        self.grid = grid

    def __call__(self):
        pass

    def update_grid(self):
        self.grid = self.__call__()
