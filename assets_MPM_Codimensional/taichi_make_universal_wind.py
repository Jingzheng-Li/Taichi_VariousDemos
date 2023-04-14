
import taichi as ti
import taichi.math as tm
import numpy as np
import taichi_math_util_funcs as ufuncs


@ti.data_oriented
class MAKE_WIND:
    def __init__(self, wind, ptclmass):
        self.wind = wind
        self.ptclmass = ptclmass


