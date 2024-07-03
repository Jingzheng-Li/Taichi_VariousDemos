
# step2 做一个fibonaci均匀取点算法 场景现在搭建好了
# https://zhuanlan.zhihu.com/p/25988652
# step3 开始写物理了 
# 问题就出在向量不知道怎么做的

import taichi as ti
import numpy as np

ti.init(arch = ti.vulkan)
#ti.init(arch = ti.cpu, debug = True)



n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1.0e-5

# Material Parameters
p_rho = 1 # density
E = 5000 # stretch
gamma = 500 # shear
k = 1000 # normal

N_Line_y, N_Line_z = 6, 6 # number of lines for y/z axis
N_Line = N_Line_y * N_Line_z # # number of total lines
line_interval = 0.01 # line space distance
ln_type2 = 200 # type2 particle count per line

# 在空间里面 这么分配start_pos会出错
start_pos = ti.Vector([0.1, 0.5, 0.5])

ln_type3 = ln_type2 - 1
n_type2 = N_Line* ln_type2
n_type3 = N_Line* ln_type3

#line length
Length = 0.8
length_type2 = Length/ (ln_type2-1)


# 边界距离
bound = 3
# 三维中旋转矩阵就不能这么写了？需要加入旋转轴了
# ROT90 = ti.Matrix([[0,-1.0],[1.0,0]]) 这个就需要按照tangent来找了
gravity = ti.Vector([0.0, -9.8, 0.0])
cf = 0.05 # 布料之间的friction

#type2
x2 = ti.Vector.field(3, dtype=float, shape=n_type2) # position 
v2 = ti.Vector.field(3, dtype=float, shape=n_type2) # velocity
C2 = ti.Matrix.field(3, 3, dtype=float, shape=n_type2) # affine velocity field
color2 = ti.Vector.field(3, dtype=float, shape=n_type2) # color 
volume2 =  dx*Length / (ln_type3+ln_type2)
mass2 = volume2 * p_rho

#type3
x3 = ti.Vector.field(3, dtype=float, shape=n_type3) # position
v3 = ti.Vector.field(3, dtype=float, shape=n_type3) # velocity
C3 = ti.Matrix.field(3, 3, dtype=float, shape=n_type3) # affine velocity field
F3 = ti.Matrix.field(3, 3, dtype=float, shape=n_type3) # deformation gradient
D3_inv = ti.Matrix.field(3, 3, dtype=float, shape=n_type3)
d3 = ti.Matrix.field(3, 3, dtype=float, shape=n_type3)
volume3 = volume2
mass3 = volume3 * p_rho

# grid
grid_v = ti.Vector.field(3, dtype= float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))
grid_f = ti.Vector.field(3, dtype= float, shape = (n_grid, n_grid, n_grid))

# ground
coordinate_origin = ti.Vector.field(3, float, shape=1)
ground = ti.Vector.field(3, float, shape=4)
ground_indices = ti.field(int, shape=2*3)
# ball
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0.5, 0, 0.5]
ball_radius = 0.25
gravity = ti.Vector([0, -9.8, 0])


@ti.func
def QR3(Mat): #3x3 mat, Gram-Schmidt Orthogonalization
    c0, c1, c2 = Mat[:,0], Mat[:,1], Mat[:,2]
    r11 = c0.norm(1e-6)
    q0 = c0/r11
    r12 = c1.dot(q0)
    q1 = c1 - r12 * q0
    r22 = q1.norm(1e-6)
    q1 /= r22
    r13 = c2.dot(q0)
    r23 = c2.dot(q1)
    q2 = c2 - r13*q0 - r23*q1
    r33 = q2.norm(1e-6)
    q2 /= r33
    Q = ti.Matrix.cols([q0,q1,q2])
    R = ti.Matrix([[r11,r12,r13],[0,r22,r23],[0,0,r33]])
    return Q, R

@ti.func
def particle_parameters_on_grid(pos):
    base = (pos * inv_dx - 0.5).cast(int) #每个顶点所在网格的左下角格点坐标
    fx = pos * inv_dx - base.cast(float) #每个格点相对左下角位置
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
    return base, fx, w

@ti.func
def GetType2FromType3(index):
    index += index // ln_type3
    return index, index+1


@ti.kernel
def initialize():

    coordinate_origin[0] = [0,0,0] # 原点坐标 参考用
    ground[0],ground[1],ground[2],ground[3] = [0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0],[1.0,0.0,1.0]
    ground_indices[0],ground_indices[1],ground_indices[2]=0,1,2
    ground_indices[3],ground_indices[4],ground_indices[5]=1,2,3

    for i in range(n_type2):
        line_num = i // ln_type2
        layer = line_num // N_Line_z # 在第z层上
        index = line_num - layer * N_Line_z # 在第z层上的第line
        x2[i] = ti.Vector([start_pos[0] + (i- line_num* ln_type2) * length_type2,
                          start_pos[1] + layer * line_interval,
                          start_pos[2] + index * line_interval])
        v2[i] = ti.math.vec3(0.0)
        C2[i] =  ti.math.mat3(0.0)
        color2[i] = (line_num/N_Line, line_num/N_Line, 1.0-line_num/N_Line)

    for i in range(n_type3):
        l, n = GetType2FromType3(i)
        x3[i] = 0.5*(x2[l] + x2[n])
        v3[i] = ti.math.vec3(0.0)
        F3[i] = ti.math.eye(3)
        C3[i] =  ti.math.mat3(0.0)


@ti.kernel
def Reset():
    for i, j, k in grid_m:
        grid_v[i,j,k] = [0.0, 0.0, 0.0]
        grid_m[i,j,k] = 0.0
        grid_f[i,j,k] = [0.0, 0.0, 0.0]

@ti.kernel
def Particle_To_Grid():
    for p in x2:
        base, fx, w = particle_parameters_on_grid(x2[p])
        affine = C2[p]
        for i, j, k in ti.static(ti.ndrange(3,3,3)):
            offset = ti.Vector([i,j,k])
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_m[base + offset] += weight * mass2
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass2 * (v2[p] +  affine@dpos)

    for p in x3:
        base, fx, w = particle_parameters_on_grid(x3[p])
        affine = C3[p]
        for i, j, k in ti.static(ti.ndrange(3,3,3)):
            offset = ti.Vector([i,j,k])
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_m[base + offset] += weight * mass3
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass3 * (v3[p] +  affine@dpos)


@ti.kernel
def Grid_Force():
    for p in x3:
        l, n = GetType2FromType3(p)
        base, fx, w = particle_parameters_on_grid(x3[p])
        dw_dx_d = ti.Matrix.rows([fx-1.5, 2*(1.0-fx), fx-0.5]) * inv_dx
        base_l, fx_l, w_l = particle_parameters_on_grid(x2[l])
        base_n, fx_n, w_n = particle_parameters_on_grid(x2[n])
        Q, R = QR3(F3[p])

        r11, r12, r13 = R[0,0], R[0,1], R[0,2]
        r22, r23, r33 = R[1,1], R[1,2], R[2,2]

        A = ti.Matrix.rows([[0,0,0],
                            [0,0,0],
                            [0,0,0]])
        
         
        #K = ti.Matrix.rows([
        #    [E*r11*(r11-1)+gamma*r12**2, gamma * r12 * r22],
        #    [gamma * r12 * r22,  -k * (1 - r22)**2 * r22 * float(r22 <= 1)]
        #    ])
        #dphi_dF = Q@ K @ R.inverse().transpose()# Q.inverse().transpose() = Q.transpose().transpose() = Q
        #dp_c1 = ti.Vector([d3[p][0,1],d3[p][1,1]])
        #dphi_dF_c1 = ti.Vector([dphi_dF[0,1],dphi_dF[1,1]])
        #Dp_inv_c0 = ti.Vector([D3_inv[p][0,0],D3_inv[p][1,0]])

        for i, j, k in ti.static(ti.ndrange(3,3,3)):
            offset = ti.Vector([i,j,k])
            weight_l = w_l[i][0] * w_l[j][1] * w_l[k][2]
            weight_n = w_n[i][0] * w_n[j][1] * w_n[k][2]

        #f_2 = dphi_dF @ Dp_inv_c0
        #grid_f[base_l + offset]+=  volume2 * weight_l * f_2
        #grid_f[base_n + offset]+= -volume2 * weight_n * f_2

        ## dphi w / x
        #dw_dx = ti.Vector([ dw_dx_d[i, 0] * w[j][1], w[i][0] * dw_dx_d[j, 1] ])
        ## technical document .(15) part 2

        #grid_f[base + offset] += -volume3 * dphi_dF_c1* dw_dx.dot( dp_c1 )



@ti.kernel
def Grid_Collision():

    for I in ti.grouped(grid_m):

        if grid_m[I] > 0:
            grid_v[I] +=  grid_f[I] * dt
            grid_v[I] /= grid_m[I]
            grid_v[I] += dt * gravity

            # seprate ball test
            dist = I*dx - ball_center[0]
            if dist.x**2 + dist.y**2 < ball_radius**2 :
                dist = dist.normalized()
                grid_v[I] -= dist * min(0, grid_v[I].dot(dist)) 
                grid_v[I] *= 0.9  #circle friction

            # sticky 
            condition = (I<bound) & (grid_v[I]<0) | (I>n_grid-bound) & (grid_v[I]>0)
            grid_v[I] = 0 if condition else grid_v[I]


@ti.kernel
def Grid_To_Particle():
    for p in x2:
        base, fx, w = particle_parameters_on_grid(x2[p])
        new_v = ti.math.vec3(0.0)
        new_C = ti.math.mat3(0.0)
        for i, j, k in ti.static(ti.ndrange(3,3,3)):
            dpos = ti.Vector([i,j,k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i,j,k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        v2[p] = new_v
        x2[p] += dt * v2[p]
        C2[p] = new_C

    # velocity and position generate from x2
    for p in x3:
        base, fx, w = particle_parameters_on_grid(x3[p])
        new_C = ti.math.mat3(0.0)
        for i, j, k in ti.static(ti.ndrange(3,3,3)):
            dpos = ti.Vector([i,j,k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i,j,k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        C3[p] = new_C


@ti.kernel
def Update_Particle_State():
    for p in x3:
        l, n = GetType2FromType3(p)
        v3[p] = 0.5 * (v2[l] + v2[n])
        x3[p] = 0.5 * (x2[l] + x2[n])

        # 这个需要修改的
        #dp1 = x2[n] - x2[l]
        #dp2 = ti.Vector([d3[p][0,1],d3[p][1,1]])
        #dp2 += dt * C3[p] @ dp2
        #d3[p] = ti.Matrix.cols([dp1,dp2])
        #F3[p] = d3[p] @ D3_inv[p]

@ti.kernel
def Return_Mapping():

    for p in x3:
        Q, R = QR3(F3[p])
        r13,r23,r33 = R[0,2], R[1,2], R[2,2]

        #if r22 < 0:
        #    r12 = 0
        #    r22 = max(r22, -1)
        #elif r22 > 1:
        #    r12 = 0
        #    r22 = 1
        #else:
        #    rr = r12**2
        #    zz = cf*(1.0 - r22)**2
        #    gamma_over_s = gamma/k
        #    f = gamma_over_s**2 * rr - zz**2
        #    if f > 0:
        #        scale = zz / ( gamma_over_s*  rr**0.5 )
        #        r12*= scale

        R[0,2],R[1,2],R[2,2] = r13,r23,r33
        F3[p] = Q @ R
        d3[p] = F3[p] @ D3_inv[p].inverse()


def main():

    initialize()

    window = ti.ui.Window('MPM Noodle', res=(1080, 1080))
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    while window.running:

        camera.position(2.0, 2.0, 3.0)
        camera.lookat(0.0, 0.0, 0.0)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.6, 0.99, 0.6))

        # return mapping可以最后再说 先把curve上面的normal找出来
        for _ in range(50):
            Reset() # √
            Particle_To_Grid() # √
            #Grid_Force()
            Grid_Collision() # √
            Grid_To_Particle() # √
            Update_Particle_State()
            #Return_Mapping()


        # 标准球在最中间
        scene.particles(ball_center,
                        radius=ball_radius,
                        color=(0.5, 0.42, 0.8))
        scene.particles(coordinate_origin,
                        radius=ball_radius * 0.05,
                        color=(0.5, 0.42, 0.8))


        scene.particles(x2, radius=ball_radius * 0.02, per_vertex_color=color2)

        # Finbanaci球 最后再说吧

        # grid mesh
        scene.mesh(ground,indices=ground_indices,two_sided=True)

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()

