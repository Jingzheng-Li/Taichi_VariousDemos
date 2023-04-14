# TODO: 暂时认为psi_coef是1.0 等到后面在加入考虑volumefraction到底是怎么回事？？

import taichi as ti
import taichi.math as tm
import numpy as np
import taichi_math_util_funcs as ufuncs

@ti.data_oriented
class MAKE_CLOTH_MEMBRANE:
    def __init__(self, vertices, ptclrestx, ptclrestvol, facerestvol, cloth_membrane_param):
        self.vertices = vertices
        self.ptclrestx = ptclrestx
        self.ptclrestvol = ptclrestvol
        self.facerestvol = facerestvol
        self.num_faces = vertices.shape[0]

        # coef_Young, coef_Poisson, coef_Visc, coef_thickness
        self.coef_Young = cloth_membrane_param[0]
        self.coef_Poisson = cloth_membrane_param[1]
        self.coef_Visc = cloth_membrane_param[2]
        self.coef_thickness = cloth_membrane_param[3]

        self.membrane_ru = ti.Vector.field(3, ti.f32, self.num_faces)
        self.membrane_rv = ti.Vector.field(3, ti.f32, self.num_faces)
        self.membrane_tensor_base = ti.Matrix([[1.0, self.coef_Poisson, 0.0],
                                        [self.coef_Poisson, 1.0, 0.0],
                                        [0.0, 0.0, 0.5 * (1.0-self.coef_Poisson)]])
        self.membrane_strtech_tensor = self.membrane_tensor_base * (self.coef_Young * self.coef_thickness / (1 - self.coef_Poisson**2))
        self.membrane_viscous_tensor = self.membrane_tensor_base * (self.coef_Visc * self.coef_thickness / (1 - self.coef_Poisson**2))

        self.psi_coeff = 1.0

    @ti.func
    def initialize_cloth_membrane(self, ):
        for f in range(self.num_faces):
            a, b, c = ufuncs.points_of_face(f, self.vertices)
            curr_DE0, curr_DE1 = self.ptclrestx[b]-self.ptclrestx[a], self.ptclrestx[c]-self.ptclrestx[a]
            # TODO: 检查交换cross是否会有问题
            curr_DE2 = tm.cross(curr_DE1, curr_DE0).normalized() # 0,1,0 
            rotmat = ufuncs.rotation_matrix_from_vectors(curr_DE2, ti.Vector([0.0,0.0,1.0]))
            uvj_3d, uvk_3d = rotmat@curr_DE0, rotmat@curr_DE1

            uvi = tm.vec2(0.0)
            uvj = ti.Vector([uvj_3d[0], uvj_3d[1]])
            uvk = ti.Vector([uvk_3d[0], uvk_3d[1]])

            dinv = 1.0 / (uvi[0] * (uvj[1] - uvk[1]) + uvj[0] * (uvk[1] - uvi[1]) + uvk[0] * (uvi[1] - uvj[1]))
            self.membrane_ru[f] = ti.Vector([dinv * (uvj[1] - uvk[1]), dinv * (uvk[1] - uvi[1]), dinv * (uvi[1] - uvj[1])])
            self.membrane_rv[f] = ti.Vector([dinv * (uvk[0] - uvj[0]), dinv * (uvi[0] - uvk[0]), dinv * (uvj[0] - uvi[0])])


    @ti.func
    def get_cloth_membrane_force(self, f, vertsid, vertspos, forcetotal):
        (a, b, c) = vertsid
        (xa, xb, xc) = vertspos
        facerestvol = self.facerestvol[f]

        vecU = xa*self.membrane_ru[f][0] + xb*self.membrane_ru[f][1] + xc*self.membrane_ru[f][2]
        vecV = xa*self.membrane_rv[f][0] + xb*self.membrane_rv[f][1] + xc*self.membrane_rv[f][2]

        # calculate inplane strain and stress
        stretch_strain = ti.Vector([0.5 * (vecU.dot(vecU) - 1.0), 0.5 * (vecV.dot(vecV) - 1.0), vecU.dot(vecV)])
        stretch_stress = self.membrane_strtech_tensor @ stretch_strain

        # point force for each vertex, gridf1 + gridf2 + gridf3 = 0
        gridforce_a = -self.psi_coeff * facerestvol * (stretch_stress[0] * (self.membrane_ru[f][0] * vecU) +
                        stretch_stress[1] * (self.membrane_rv[f][0] * vecV) +
                        stretch_stress[2] * (self.membrane_ru[f][0] * vecV + self.membrane_rv[f][0] * vecU))
        gridforce_b = -self.psi_coeff * facerestvol * (stretch_stress[0] * (self.membrane_ru[f][1] * vecU) +
                        stretch_stress[1] * (self.membrane_rv[f][1] * vecV) +
                        stretch_stress[2] * (self.membrane_ru[f][1] * vecV + self.membrane_rv[f][1] * vecU))
        gridforce_c = -self.psi_coeff * facerestvol * (stretch_stress[0] * (self.membrane_ru[f][2] * vecU) +
                        stretch_stress[1] * (self.membrane_rv[f][2] * vecV) +
                        stretch_stress[2] * (self.membrane_ru[f][2] * vecV + self.membrane_rv[f][2] * vecU))

        forcetotal[a] += gridforce_a
        forcetotal[b] += gridforce_b
        forcetotal[c] += gridforce_c

        # calculate in plane viscous
        # viscxa, viscxb, viscxc = ptclx[a], ptclx[b], ptclx[c]
        # viscVecU = viscxa*membrane_ru[f][0] + viscxb*membrane_ru[f][1] + viscxc*membrane_ru[f][2]
        # viscVecV = viscxa*membrane_rv[f][0] + viscxb*membrane_rv[f][1] + viscxc*membrane_rv[f][2]
        # viscous_strain = ti.Vector([0.5 * (VecU.dot(VecU) -viscVecU.dot(viscVecU)), 
        #                             0.5 * (VecV.dot(VecV) - viscVecV.dot(viscVecV)), 
        #                             VecU.dot(VecV) - viscVecV.dot(viscVecV)])
        # viscous_stress = membrane_viscous_tensor @ viscous_strain


    # 这个psi是什么？？
    # TODO：这个地方果然有问题 Triplet天生就是三元组 就是用来 而且这个地方也没有放到grid里面
    @ti.func
    def get_cloth_membrane_Hessian(self, f, vertsid, vertspos, Hesstotal):
        (a, b, c) = vertsid
        (xa, xb, xc) = vertspos

        vecU = xa*self.membrane_ru[f][0] + xb*self.membrane_ru[f][1] + xc*self.membrane_ru[f][2]
        vecV = xa*self.membrane_rv[f][0] + xb*self.membrane_rv[f][1] + xc*self.membrane_rv[f][2]

        for i,j in ti.static(ti.ndrange(3,3)):
            dphidx_membrane =((self.psi_coeff * self.facerestvol[f]) *
            (self.membrane_strtech_tensor[0,0] * (self.membrane_ru[f][i] * self.membrane_ru[f][j] * vecU.outer_product(vecU)) +
            self.membrane_strtech_tensor[1,1] * (self.membrane_rv[f][i] * self.membrane_rv[f][j] * vecV.outer_product(vecV)) +
            self.membrane_strtech_tensor[2,2] * ((self.membrane_rv[f][i] * self.membrane_ru[f][j] * vecU.outer_product(vecV) +
                                                self.membrane_ru[f][i] * self.membrane_rv[f][j] * vecV.outer_product(vecU)) +
                                                (self.membrane_rv[f][i] * self.membrane_rv[f][j] * vecU.outer_product(vecU) +
                                                self.membrane_ru[f][i] * self.membrane_ru[f][j] * vecV.outer_product(vecV))) +
            self.membrane_strtech_tensor[0,1] * (self.membrane_ru[f][i] * self.membrane_rv[f][j] * vecU.outer_product(vecV)) +
            self.membrane_strtech_tensor[1,0] * (self.membrane_rv[f][i] * self.membrane_ru[f][j] * vecV.outer_product(vecU))) )

            for r,s in ti.static(ti.ndrange(3,3)):
                Hesstotal[f,i,j][s,r] = dphidx_membrane[s,r]


    @ti.func
    def get_cloth_membrane_multiplier(self, ):
        pass




