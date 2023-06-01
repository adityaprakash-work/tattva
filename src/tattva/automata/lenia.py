# ---INFO-----------------------------------------------------------------------

# --Authors        |Last Updated |tattva-automata-lenia
# --Aditya Prakash |17-05-2023   |stable

# ---DEPENDENCIES---------------------------------------------------------------
import jax.numpy as jnp
from jax import jit

from ..core import layers as ttvl
from ..utils import maths
from .base import BaseAutomaton


# --- GENERALIZED LENIA---------------------------------------------------------
class NDLenia(BaseAutomaton):
    def __init__(self, grid_0: jnp.array, dt: float = 0.1):
        super().__init__()
        self.kernels = jnp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).reshape(
            (1, 3, 3, 1)
        )
        self.init_grid(grid_0)
        self.dt = dt
        self.growth_func = jit(lambda U: 0 + (U == 3) - ((U < 2) | (U > 3)))
        self.call_dict = {
            "cpadding": ttvl.CircularPadding((self.kernels[0].shape)),
            "cgolrule": ttvl.Potential(self.kernels, method="direct"),
            "cgolgrow": ttvl.Growth(self.growth_func, self.dt, maths.hardclip),
        }

    def __call__(self):
        pgrid = self.call_dict["cpadding"](self.grid)
        pdist = self.call_dict["cgolrule"](pgrid)
        ngrid = self.call_dict["cgolgrow"](pdist)
        return ngrid
