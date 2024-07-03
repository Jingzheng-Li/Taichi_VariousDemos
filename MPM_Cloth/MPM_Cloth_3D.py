
import taichi as ti
import numpy as np

ti.init(arch = ti.vulkan)
#ti.init(arch = ti.cpu, debug = True)

n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 5.0e-5

# Material Parameters
p_rho = 1 # density
E = 5000 # Young's modulus
nu = 0.0 # poisson
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)) # lambda = 0
gamma = 500 # shear stiffness
kb = 1000 # bending stiffness
cf = 0.05 # friction between cloth
cff = 0.1 # friction between ground

length_cloth = 0.8
start_pos_cloth = [0.1, 0.5, 0.1]
n_square = n_grid
quad_size = length_cloth / n_square
n_triangles = (n_square - 1) * (n_square - 1) * 2

bound = 3
gravity = ti.Vector([0.0, -9.8, 0.0])

#type2
n_type2 = n_square * n_square
x2 = ti.Vector.field(3, float, shape=n_type2) # position 
v2 = ti.Vector.field(3, float, shape=n_type2) # velocity
C2 = ti.Matrix.field(3, 3, float, shape=n_type2) # affine
indices2 = ti.field(int, shape=n_triangles * 3)
color2 = ti.Vector.field(3, float, shape=n_type2)


n_type3 = n_triangles
x3 = ti.Vector.field(3, dtype=float, shape=n_type3) # position
v3 = ti.Vector.field(3, dtype=float, shape=n_type3) # velocity
C3 = ti.Matrix.field(3, 3, dtype=float, shape=n_type3) # affine velocity field
F3 = ti.Matrix.field(3, 3, dtype=float, shape=n_type3) # deformation gradient
D3_inv = ti.Matrix.field(3, 3, dtype=float, shape=n_type3)
D3 = ti.Matrix.field(3, 3, dtype=float, shape=n_type3)

# triangle initial parameter
volume = ti.field(float, shape=n_triangles)
mass = ti.field(float, shape=n_triangles)

# grid
grid_v = ti.Vector.field(3, dtype= float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))
grid_f = ti.Vector.field(3, dtype= float, shape = (n_grid, n_grid, n_grid))

# ground
coordinate_origin = ti.Vector.field(3, float, shape=1)
ground = ti.Vector.field(3, float, shape=4)
ground_indices = ti.field(int, shape=2*3)
# ball
ball_center = ti.Vector.field(3, dtype=float, shape=1)
ball_center[0] = [0.5, 0, 0.5]
ball_radius = 0.1
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


# 这个地方对x2和x3都要产生一系列parameter
@ti.func
def particle_parameters_on_grid(pos):
    base = (pos * inv_dx - 0.5).cast(int) 
    fx = pos * inv_dx - base.cast(float) 
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
    return base, fx, w


@ti.func
def GetType2FromType3(index, particle):
    para1 = particle[indices2[3*index]]
    para2 = particle[indices2[3*index+1]]
    para3 = particle[indices2[3*index+2]]
    return para1, para2, para3

@ti.kernel
def initialize():

    coordinate_origin[0] = [0,0,0] # 原点坐标 参考用
    ground[0],ground[1] = [0.0,0.0,0.0],[1.0,0.0,0.0]
    ground[2],ground[3] = [0.0,0.0,1.0],[1.0,0.0,1.0]
    ground_indices[0],ground_indices[1],ground_indices[2]=0,1,2
    ground_indices[3],ground_indices[4],ground_indices[5]=1,2,3

    for i, j in ti.ndrange(n_square, n_square):
        x2[i * n_square + j] = [start_pos_cloth[0] + i*quad_size,
                                start_pos_cloth[1],
                                start_pos_cloth[2] + j*quad_size]
        v2[i * n_square + j] = ti.math.vec3(0.0)
        C2[i * n_square + j] = ti.math.mat3(0.0)
        color2[i * n_square + j] = [0.6, 1.0, 0.6]

    for i, j in ti.ndrange(n_square - 1, n_square - 1):
        quad_id = (i * (n_square - 1)) + j
        # 1st triangle of the square
        indices2[quad_id * 6 + 0] = i * n_square + j
        indices2[quad_id * 6 + 1] = (i + 1) * n_square + j
        indices2[quad_id * 6 + 2] = i * n_square + (j + 1)
        # 2nd triangle of the square
        indices2[quad_id * 6 + 3] = (i + 1) * n_square + j + 1
        indices2[quad_id * 6 + 4] = i * n_square + (j + 1)
        indices2[quad_id * 6 + 5] = (i + 1) * n_square + j

    for i in range(n_triangles):
        pos1, pos2, pos3 = GetType2FromType3(i, x2)
        x3[i] = (pos1 + pos2 + pos3) / 3.0
        v3[i] = ti.math.vec3(0.0)
        F3[i] = ti.math.eye(3)
        C3[i] =  ti.math.mat3(0.0)

        tri_area = (ti.math.cross(pos2-pos1, pos3-pos1)).norm()
        volume[i] = 0.5 * tri_area
        mass[i] = volume[i] * p_rho

        dp0, dp1 = pos2-pos1, pos3-pos1
        dp2 = (ti.math.cross(dp0, dp1)).normalized(1e-6)
        D3[i] = ti.Matrix.cols([dp0,dp1,dp2])
        D3_inv[i] = D3[i].inverse()   


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
            grid_m[base + offset] += weight * mass[0]
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass[0] * (v2[p] +  affine @ dpos)

    for p in x3:
        base, fx, w = particle_parameters_on_grid(x3[p])
        affine = C3[p]
        for i, j, k in ti.static(ti.ndrange(3,3,3)):
            offset = ti.Vector([i,j,k])
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_m[base + offset] += weight * mass[0]
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass[0] * (v3[p] +  affine@dpos)

# 这个pos到底对不对？pos好像有点问题 就剩这一个grid_force了？
# 完全不知道怎么回事？更新完grid force就对了
@ti.kernel
def Grid_Force():
    for i in range(n_triangles):
        pos1, pos2, pos3 = GetType2FromType3(i, x2)
        base, fx, w = particle_parameters_on_grid(x3[i])
        dw_dx_d = ti.Matrix.rows([fx-1.5, 2*(1.0-fx), fx-0.5]) * inv_dx

        base_1, fx_1, w_1 = particle_parameters_on_grid(pos1)
        base_2, fx_2, w_2 = particle_parameters_on_grid(pos2)
        base_3, fx_3, w_3 = particle_parameters_on_grid(pos3)

        Q, R = QR3(F3[i])
        r13, r23, r33 = R[0,2], R[1,2], R[2,2]
        R_hat = ti.math.mat2([R[0,0], R[0,1]],[0.0, R[1,1]])
        r_hat = ti.math.vec2(R[0,2], R[1,2])
        F_hat = ti.math.mat2([F3[i][0,0], F3[i][0,1]],[0.0, F3[i][1,1]])

        f_deriv = -kb * (1-r33)**2 * float(r33 <= 1);
        g_deriv = gamma * (r13 + r23);

        trace = (R_hat.transpose() @ F_hat - ti.math.eye(2)).trace()
        PK1 = 2*mu*(F_hat-R_hat) + la*trace*R_hat
        A11 = PK1 @ R_hat.transpose() + g_deriv * r_hat @ r_hat.transpose()
        A12 = g_deriv * r33 * r_hat
        A22 = f_deriv * r33

        A = ti.math.mat3([A11[0,0],A11[0,1],A12[0]],
                         [A11[1,0],A11[1,1],A12[1]],
                         [A12[0], A12[1], A22])

        dphi_dF = Q @ A @ R.inverse().transpose()

        dp_c2 = ti.Vector([D3[i][0,2],D3[i][1,2],D3[i][2,2]])
        dphi_dF_c2 = ti.Vector([dphi_dF[0,2],dphi_dF[1,2],dphi_dF[2,2]])
        Dp_inv_c0 = ti.Vector([D3_inv[i][0,0],D3_inv[i][1,0],D3_inv[i][2,0]])

        #for i, j, k in ti.static(ti.ndrange(3,3,3)):
        #    offset = ti.Vector([i,j,k])
        #    weight_1 = w_1[i][0] * w_1[j][1] * w_1[k][2]
        #    weight_2 = w_2[i][0] * w_2[j][1] * w_2[k][2]
        #    weight_3 = w_3[i][0] * w_3[j][1] * w_3[k][2]

        #    f_2 = dphi_dF @ Dp_inv_c0
        #    grid_f[base_1 + offset]+=  volume[i] * weight_1 * f_2
        #    grid_f[base_2 + offset]+= -volume[i] * weight_2 * f_2
        #    grid_f[base_3 + offset]+= -volume[i] * weight_3 * f_2

        #    # dphi w / x 这个很可能有问题?
        #    dw_dx = ti.Vector([w[k][2] * dw_dx_d[i, 0], 
        #                       w[j][1] * dw_dx_d[j, 1],
        #                       w[i][0] * dw_dx_d[k, 2] ])


        #    # technical document .(15) part 2
        #    grid_f[base + offset] += -volume[i] * dphi_dF_c2* dw_dx.dot(dp_c2)




@ti.kernel
def Grid_Collision():
    for I in ti.grouped(grid_m):

        if grid_m[I] > 0:
            grid_v[I] +=  grid_f[I] * dt
            grid_v[I] /= grid_m[I]
            grid_v[I] += dt * gravity

            # seprate ball test
            dist = I*dx - ball_center[0]
            if dist.x**2 + dist.y**2 < (ball_radius+dx)**2:
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
    for i in range(n_triangles):
        pos1, pos2, pos3 = GetType2FromType3(i, x2)
        vel1, vel2, vel3 = GetType2FromType3(i, v2)
        x3[i] = (pos1 + pos2 + pos3) / 3.0
        v3[i] = (vel1 + vel2 + vel3) / 3.0

        dp0, dp1 = pos2-pos1, pos3-pos1
        dp2 = ti.math.vec3(D3[i][0,2], D3[i][1,2], D3[i][2,2])
        dp2 += dt * C3[i] @ dp2
        D3[i] = ti.Matrix.cols([dp0,dp1,dp2])
        F3[i] = D3[i] @ D3_inv[i]

  
@ti.kernel
def Return_Mapping():
    for i in range(n_triangles):
        Q, R = QR3(F3[i])
        r13, r23, r33 = R[0,2], R[1,2], R[2,2]
        r_hat = ti.math.vec2(r13, r23)
        r_hat_norm = r_hat.norm(1e-8)
        if r33 > 1.0:
            r33 = 1.0
            r13, r23 = 0.0, 0.0
        elif r33<0.0:
            r13, r23, r33 = 0.0, 0.0, 0.0
        else:            
            f = (gamma/kb)*r_hat_norm - cf*((1.0-r33)**2)
            if f > 0.0:
                scale = (kb/gamma)*cf*((1.0-r33)**2)/r_hat_norm
                r13 *= scale
                r23 *= scale

        R[0,2], R[1,2], R[2,2] = r13, r23, r33
        F3[i] = Q @ R
        D3[i] = F3[i]@D3_inv[i].inverse()

def main():

    initialize()

    window = ti.ui.Window('MPM Noodle', res=(1080, 1080))
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    while window.running:

        camera.position(-1.0, 2.0, 2.0)
        camera.lookat(0.5, 0.0, 0.5)
        scene.set_camera(camera)
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((1.0, 1.0, 1.0))
                
        
        for _ in range(50):
            Reset() # √
            Particle_To_Grid() # √
            Grid_Force()
            Grid_Collision() # √
            Grid_To_Particle() # √
            Update_Particle_State() # √
            Return_Mapping() # √


        # ball and ground
        scene.particles(ball_center,
                        radius=ball_radius,
                        color=(0.5, 0.42, 0.8))
        scene.particles(coordinate_origin,
                        radius=ball_radius * 0.25,
                        color=(0.5, 0.42, 0.8))
        scene.mesh(ground,indices=ground_indices,two_sided=True)

        # mesh2 and 3
        scene.mesh(x2, indices=indices2,
                   per_vertex_color=color2,
                   two_sided=True)
        #scene.particles(x3, radius=ball_radius * 0.005,
        #                color=(1.0, 0.0, 0.0))


        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()

