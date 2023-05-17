# ---INFO-----------------------------------------------------------------------

# --Authors        |Last Updated |tattva-automata-cgol (Conway's game of life)
# --Aditya Prakash |17-05-2023   |stable

# ---DEPENDENCIES---------------------------------------------------------------
import jax.numpy as jnp
from jax import jit

from .base import BaseAutomaton
from ..core import layers as ttvl
from ..utils import maths


# ---CONWAY'S GAME OF-----------------------------------------------------------
class ConwayGOL(BaseAutomaton):
    def __init__(self, grid_shape: tuple, grid_0: jnp.array, dt: float):
        super().__init__(grid_shape)
        self.kernel = jnp.array([[[1], [1], [1]], [[1], [0], [1]], [[1], [1], [1]]])
        self.init_grid(grid_0)
        self.dt = 0.1
        self.growth_func = jit(lambda U: 0 + (U == 3) - ((U < 2) | (U > 3)))
        self.call_dict = {
            "cpadding": ttvl.CircularPadding((self.kernel.shape)),
            "cgolrule": ttvl.Potential(self.kernel, method="direct"),
            "cgolgrow": ttvl.Growth(self.growth_func, self.dt, maths.hardclip),
        }

    def __call__(self):
        pgrid = self.call_dict["cpadding"](self.grid)
        pdist = self.call_dict["cgolrule"](pgrid)
        ngrid = self.call_dict["cgolgrow"](pdist)
        return ngrid
