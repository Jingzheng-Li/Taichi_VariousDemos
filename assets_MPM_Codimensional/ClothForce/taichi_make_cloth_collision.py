# TODO：normal计算太重了 总感觉normal计算这里有点小问题？？

import taichi as ti
import taichi.math as tm
import numpy as np
import taichi_math_util_funcs as ufuncs

@ti.data_oriented
class MAKE_CLOTH_COLLISION:
    def __init__(self, vertices, ptclrestx, ptclrestvol, facerestvol, cloth_collision_param):
        self.vertices = vertices
        self.ptclrestx = ptclrestx
        self.ptclrestvol = ptclrestvol
        self.facerestvol = facerestvol
        self.num_faces = vertices.shape[0]
        
        self.coef_normalstiff = cloth_collision_param[0]
        self.coef_shearstiff = cloth_collision_param[1]
        self.coef_fric = cloth_collision_param[2]

        self.faceF = ti.Matrix.field(3, 3, ti.f32, self.num_faces) # deformation gradient
        self.faceDE = ti.Matrix.field(3, 3, ti.f32, self.num_faces) # intial material directions(1:tangent, 2:orthogonal)
        self.faceDE_inv = ti.Matrix.field(3, 3, ti.f32, self.num_faces)
        self.facedE = ti.Matrix.field(3, 3, ti.f32, self.num_faces) # deformed elastic material directions

    @ti.func
    def initialize_cloth_collision(self, ):
        for f in range(self.num_faces):
            a, b, c = ufuncs.points_of_face(f, self.vertices)
            DE0, DE1 = self.ptclrestx[b]-self.ptclrestx[a], self.ptclrestx[c]-self.ptclrestx[a] # fem mesh part
            DE1_cross_DE0 = tm.cross(DE1, DE0)
            DE2 = DE1_cross_DE0.normalized() # norm: 0.0,1.0,0.0
            MatDE = ti.Matrix.cols([DE0, DE1, DE2])

            # from norm to axis-z
            rotmat1 = ufuncs.rotation_matrix_from_vectors(DE2, ti.Vector([0.0,0.0,1.0]))
            rotu, rotv = rotmat1 @ DE0, rotmat1 @ DE1
            # from t0 to axis-x 
            rotmat2 = ufuncs.rotation_matrix_from_vectors(rotu, ti.Vector([1.0,0.0,0.0]))
            proju, projv = rotmat2 @ rotu, rotmat2 @ rotv

            MatDEproj = tm.eye(3)
            MatDEproj[0,0] = proju[0]
            MatDEproj[0,1] = projv[0]
            MatDEproj[1,1] = projv[1]
            invDstar = MatDEproj.inverse()

            self.facedE[f] = MatDE
            self.faceDE[f] = MatDEproj
            self.faceDE_inv[f] = invDstar
            self.faceF[f] = MatDE @ invDstar

        
    @ti.func
    def get_symmetric_A(self, R):
        # dPsi_dR = [P_hat, g'r | 0, f']
        dPsidR = tm.mat3(0)

        dhdr13, dhdr23, dhdr33 = 0.0, 0.0, 0.0
        r13, r23, r33 = R[0,2], R[1,2], R[2,2]
        if r33 <= 1.0:
            dhdr33 = -self.coef_normalstiff * (1.0 - r33)**2
        else:
            dhdr33 = 0.0
        if dhdr33 != 0.0:
            dhdr13 = self.coef_shearstiff * r13
            dhdr23 = self.coef_shearstiff * r23

        dPsidR[0,2] = dhdr13
        dPsidR[1,2] = dhdr23
        dPsidR[2,2] = dhdr33

        dPsidR_RT = dPsidR @ R.transpose()
        upper_Mat = ti.Matrix([[0.0, dPsidR_RT[0,1], dPsidR_RT[0,2]],
                            [0.0, 0.0, dPsidR_RT[1,2]],
                            [0.0, 0.0, 0.0]])
        diag_Mat = ti.Matrix([[dPsidR_RT[0,0], 0.0, 0.0],
                            [0.0, dPsidR_RT[1,1], 0.0],
                            [0.0, 0.0, dPsidR_RT[2,2]]])
        lower_Mat = upper_Mat.transpose()
        sym_A = upper_Mat + lower_Mat + diag_Mat
        return sym_A

    @ti.func
    def get_cloth_collision_force(self, f, ):

        Q, R = ufuncs.QR3(self.faceF[f])
        sym_A = self.get_symmetric_A(R)
        dPsi_dF = Q @ sym_A @ R.inverse().transpose()
        dPsi_dF2 = ti.Vector([dPsi_dF[0,2],dPsi_dF[1,2],dPsi_dF[2,2]]) # dPsi/dF第三列orthogonal部分
        dE2 = ti.Vector([self.facedE[f][0,2],self.facedE[f][1,2],self.facedE[f][2,2]]) # dE第三列orthogonal部分

        return dPsi_dF2, dE2


    @ti.func
    def get_cloth_return_mapping(self, f, ptclx, dt, faceC):

        a, b, c = ufuncs.points_of_face(f, self.vertices)
        dE0, dE1 = ptclx[b]-ptclx[a], ptclx[c]-ptclx[a] # dE第一列tangent方向
        dE2 = ti.Vector([self.facedE[f][0,2],self.facedE[f][1,2],self.facedE[f][2,2]])
        dE2 = (tm.eye(3) + dt * faceC[f]) @ dE2
        self.facedE[f] = ti.Matrix.cols([dE0,dE1,dE2])
        self.faceF[f] = self.facedE[f] @ self.faceDE_inv[f]

        Q, R = ufuncs.QR3(self.faceF[f])
        r33 = R[2,2]
        r13, r23 = R[0,2], R[1,2]
        r13_r23 = ti.Vector([r13, r23])

        if r33 < 1.0:
            # return mapping: scale*coefshear*|r| < coeffric*coefstiff*(r33-1)^2
            fn = self.coef_shearstiff * r13_r23.norm(1e-8)
            ff = self.coef_normalstiff * (1.0 - r33)**2
            if ff > self.coef_fric * fn:
                r13 *= ti.min(1.0, self.coef_fric * fn / ff)
                r23 *= ti.min(1.0, self.coef_fric * fn / ff)
        else:
            r13 = 0.0
            r23 = 0.0
            r33 = 1.0

        R[0,2], R[1,2], R[2,2] = r13, r23, r33

        self.faceF[f] = Q @ R
        self.facedE[f] = self.faceF[f] @ self.faceDE[f]


