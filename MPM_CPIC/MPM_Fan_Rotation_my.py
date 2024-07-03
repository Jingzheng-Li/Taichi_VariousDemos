

import taichi as ti
import numpy as np

# 不对啊 没有数组越界？
ti.init(arch=ti.vulkan) # Try to run on GPU
#ti.init(arch=ti.cpu, debug=True)


quality = 3 # Use a larger value for higher-res simulations
n_particles, n_grid = 10000 * quality ** 2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters
gravity = ti.math.vec2(0.0,-2.8)

# elastic particles
x = ti.Vector.field(2, dtype=float, shape=n_particles) # position
v = ti.Vector.field(2, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id
Jp = ti.field(dtype=float, shape=n_particles) # plastic deformation

# rigid surface
r_v = 1 # velocity of the rigid surface
n_rseg = 100 # num of rigid segments
x_r = ti.Vector.field(2, dtype=float, shape=n_rseg+1) # location of nodes on the rigid surface
x_rp = ti.Vector.field(2, dtype=float, shape=n_rseg) # location of rigid particles
x_ls = [0.8, 0.5]
x_le = [1.2, 0.6]

grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid)) # grid node mass

# 这个grid显示solid本身所占的区域
grid_d = ti.field(dtype=float, shape=(n_grid, n_grid))
grid_A = ti.field(dtype=int, shape=(n_grid, n_grid))
grid_T = ti.field(dtype=int, shape=(n_grid, n_grid)) # grid_tag

# 这个会显示和solid接触的部分例子
p_d = ti.field(dtype=float, shape=n_particles)
p_A = ti.field(dtype=int, shape=n_particles)
p_T = ti.field(dtype=int, shape=n_particles) # particle_tag
p_n = ti.Vector.field(2, dtype=float, shape=n_particles)

energy = ti.field(dtype=float, shape=())

@ti.kernel
def initialize():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.7 + 0.3]
        v[i] = ti.math.vec2(0.0)
        F[i] = ti.math.eye(2)
        Jp[i] = 1.0

    x_r[0] = x_ls
    for i in range(n_rseg):
        x_r[i+1] = [x_ls[0] + (x_le[0]-x_ls[0]) / n_rseg * (i+1), x_ls[1] + (x_le[1]-x_ls[1]) / n_rseg * (i+1)]
        x_rp[i] = (x_r[i] + x_r[i+1]) / 2

# 其中有几个能够合并 尽量减少便利的过程
@ti.kernel
def reset_grid():
    for i,j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

        grid_A[i,j] = 0
        grid_T[i,j] = 0
        grid_d[i,j] = 0.0

    for p in x:
        p_A[p] = 0
        p_T[p] = 0
        p_d[p] = 0.0

@ti.kernel
def update_solid_line():
    # 更新solid line的位置
    line = ti.Vector([x_le[0]-x_ls[0],x_le[1]-x_ls[1]]).normalized()
    for p in x_r:
        x_r[p] = x_r[p] - line * dt * r_v

    for p in x_rp:
        x_rp[p] = x_rp[p] - line * dt * r_v


@ti.kernel
def update_solid_area():
    for p in x_rp:
        # solid两个点 
        ba = x_r[p+1] - x_r[p]
        # solid在p点的base
        base = (x_rp[p] * inv_dx - 0.5).cast(int)
        for i,j in ti.static(ti.ndrange(3,3)):
            offset = ti.math.ivec2(i,j)
            pa = (offset + base).cast(float) * dx - x_r[p] # p和格点的向量
            h = pa.dot(ba) / (ba.dot(ba)) # 计算格点投影向量投影长度
            if h>=0.0 and h<=1.0:
                grid_d[base+offset] = (pa - h*ba).norm() # 格点到solid长度
                grid_A[base+offset] = 1 # 符合solid的范围
                outer = ti.math.cross(pa, ba) # 判断点在solid的那一侧
                # 获得符号距离场
                if outer > 0:
                  grid_T[base + offset] = 1
                else:
                  grid_T[base + offset] = -1


    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        Tpr = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            # 如果这个点在solid三角性范围内
            if grid_A[base + offset] == 1:
                p_A[p] = 1 # 就取particle也为1

            weight = w[i][0] * w[j][1] # 合适的weight
            Tpr += weight * grid_d[base + offset] * grid_T[base + offset] #到格点长度的加权和

        # 如果确信式在solid的影响区域之内
        if p_A[p] == 1:
            # 判断在哪一侧
            if Tpr > 0:
                p_T[p] = 1
            else:
                p_T[p] = -1
            p_d[p] = abs(Tpr) # 一样获得到格点的长度 



@ti.kernel
def particle_to_grid():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]

        h = 0.5
        mu, la = mu_0 * h, lambda_0 * h
        U, sig, V = ti.svd(F[p])
        J = 1.0
        J *= sig[0, 0] * sig[1, 1]
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])

            # 如果网格和particle根本不在一侧 就完全不care
            if p_T[p] * grid_T[base+offset] == -1:
                pass
            else:
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass


@ti.kernel
def update_state():
    for i, j in grid_m:
        if grid_m[i, j] > 0: # No need for epsilon here
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j] # Momentum to velocity
            grid_v[i, j] += dt * gravity # gravity
      
            # seperate boundary      
            if i < 3 and grid_v[i, j][0] < 0:          grid_v[i, j][0] = 0 # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:          grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0


# 这个G2P的过程不需要
@ti.kernel
def grid_to_particle():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            g_v = ti.Vector([0.0, 0.0])

            # 还是 如果particle和网格不在同一侧
            if p_T[p] * grid_T[base + ti.Vector([i,j])] == -1:
                # 把line变成向量 为啥要这么写？直接写向量不行吗？取过normalized了
                line = ti.Vector([x_le[0]-x_ls[0],x_le[1]-x_ls[1]]).normalized()
                # 获得点到line开头的向量
                pa = ti.Vector([x[p][0]-x_ls[0], x[p][1]-x_ls[1]])

                # 处理particle和rigid body的边界条件 pa扔掉在line上的投影
                np = (pa - pa.dot(line)*line).normalized() #separate
                sg = v[p].dot(np)
                # 速度投影
                if sg > 0:
                  g_v = v[p]
                else:
                  g_v = v[p].dot(line) * line

            else:
                g_v = grid_v[base + offset]

            dpos = offset.cast(float) - fx
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p] # advection


initialize()
window = ti.ui.Window("Taichi MLS-MPM-Cutting", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1.0,1.0,1.0))

# 根本没有显示出来东西为什么？
while window.running:

    canvas.circles(x, radius=0.003, color=(0, 0.5, 0.5))
    canvas.circles(x_r, radius=0.003, color=(0.6, 1.0, 0.6))

    # 原来从MPM到CPIC就只差几行代码 此外还需要有一个CDF
    for s in range(int(5e-3 // dt)):
        reset_grid()
        update_solid_line()
        update_solid_area()
        particle_to_grid()
        update_state()
        grid_to_particle()
        
    window.show()