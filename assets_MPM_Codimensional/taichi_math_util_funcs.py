
import taichi as ti
import taichi.math as tm
import numpy as np

@ti.func
def points_of_face(faceid, vertices):
    vert1 = vertices[faceid].x
    vert2 = vertices[faceid].y
    vert3 = vertices[faceid].z
    return vert1, vert2, vert3


@ti.func
def QR3(Mat):
    c0 = ti.Vector([Mat[0,0],Mat[1,0],Mat[2,0]])
    c1 = ti.Vector([Mat[0,1],Mat[1,1],Mat[2,1]])
    c2 = ti.Vector([Mat[0,2],Mat[1,2],Mat[2,2]])
    q0 = c0.normalized(1e-8)
    q1 = c1 - c1.dot(q0)*q0
    q1 = q1.normalized(1e-8)
    q2 = c2 - c2.dot(q0)*q0 - c2.dot(q1)*q1
    q2 = q2.normalized(1e-8)
    Q = ti.Matrix.cols([q0,q1,q2])
    R = Q.inverse() @ Mat
    return Q, R


@ti.func
def split_sym_skew(Mat):
    Mat_trans = Mat.transpose()
    Mat_sym = 0.5 * (Mat + Mat_trans)
    Mat_skew = 0.5 * (Mat - Mat_trans)
    return Mat_sym, Mat_skew


@ti.func
def B_spline_weight(ptcl_pos, inv_dx):
    base = (ptcl_pos * inv_dx - tm.vec3(0.5)).cast(int) # point在grid的m行n列 和3*3grid的左下角base 3dim vector
    fx = ptcl_pos * inv_dx - tm.vec3(base) # 距离3*3grid base的距离 3dim vector
    w = tm.mat3(0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2) # 能够对周围3*3影响的weight 3*3dim matrix
    return base, fx, w


# rotation matrix of vec1 rotate to vec2
@ti.func
def rotation_matrix_from_vectors(vec1, vec2):
    vec1_normal = vec1.normalized()
    vec2_normal = vec2.normalized()
    vec = tm.cross(vec1_normal, vec2_normal)    
    rotmat = tm.eye(3)
    if vec.norm() > 1e-8:
        vec_dot = tm.dot(vec1_normal, vec2_normal)
        vec_norm = vec.norm()
        kmat = ti.Matrix([[0, -vec[2], vec[1]],[vec[2], 0, -vec[0]],[-vec[1], vec[0], 0]])
        rotmat = tm.eye(3) + kmat + kmat@kmat * ((1 - vec_dot) / (vec_norm ** 2))
    return rotmat