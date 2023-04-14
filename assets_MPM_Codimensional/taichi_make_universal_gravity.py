
import taichi as ti
import taichi.math as tm
import numpy as np
import taichi_math_util_funcs as ufuncs

@ti.data_oriented
class MAKE_UNIVERSAL_GRAVITY:
    def __init__(self, gravity, ptclmass):
        self.gravity = gravity
        self.ptclmass = ptclmass

    @ti.func
    def get_liquid_gravity_force(self, ):
        pass

    @ti.func
    def get_gravity_force(self, p, forcetotal):
        forcetotal[p] += self.gravity * self.ptclmass[p]












