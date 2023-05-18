# ---INFO-----------------------------------------------------------------------

# --Authors        |Last Updated |tattva-automata-lenia
# --Aditya Prakash |17-05-2023   |stable

# ---DEPENDENCIES---------------------------------------------------------------
import jax.numpy as jnp

from .base import BaseAutomaton


# --- GENERALIZED LENIA---------------------------------------------------------
class Lenia(BaseAutomaton):
    def __init__(self, grid_shape: tuple):
        super(Lenia, self).__init__(grid_shape)

