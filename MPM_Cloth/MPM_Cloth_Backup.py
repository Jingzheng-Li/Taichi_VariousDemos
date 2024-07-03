

# 先在上面按照自己想法修改简化
# 先修改成3D版本 
# 真的有错误


import taichi as ti
import numpy as np

ti.init(arch = ti.vulkan)

Circle_Center = ti.Vector([0.7, 0.2])
Circle_Radius = 0.4

dim = 2
n_grid = 256
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 4.0e-5

p_rho = 1

# Material Parameters
E = 5000 # stretch
gamma = 500 # shear
k = 1000 # normal

# number of lines
N_Line = 12
# line space distance
line_interval = 0.009
# type2 particle count per line
ln_type2 = 200

start_pos = ti.Vector([0.2, 0.8])

ln_type3 = ln_type2 - 1
n_type2 = N_Line* ln_type2
n_type3 = N_Line* ln_type3

#line length
Length = 0.75
length_type2 = Length/ (ln_type2-1) # 每个2--3--2的长度

#type2
x2 = ti.Vector.field(2, dtype=float, shape=n_type2) # position 
v2 = ti.Vector.field(2, dtype=float, shape=n_type2) # velocity
C2 = ti.Matrix.field(2, 2, dtype=float, shape=n_type2) # affine velocity field
volume2 =  dx*Length / (ln_type3+ln_type2)

#type3
x3 = ti.Vector.field(2, dtype=float, shape=n_type3) # position
v3 = ti.Vector.field(2, dtype=float, shape=n_type3) # velocity
C3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3) # affine velocity field
F3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3) # deformation gradient
D3_inv = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)
D3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)
volume3 = volume2


grid_v = ti.Vector.field(2, dtype= float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))
grid_f = ti.Vector.field(2, dtype= float, shape = (n_grid, n_grid))

n_segment = n_type3

ROT90 = ti.Matrix([[0,-1.0],[1.0,0]])

mass = volume2 * p_rho
bound = 3

@ti.func
def QR2(Mat): #2x2 mat, Gram–Schmidt Orthogonalization
    c0, c1 = Mat[:,0], Mat[:,1]
    r11 = c0.norm(1e-6)
    q0 = c0/r11
    r12 = c1.dot(q0)
    q1 = c1 - r12 * q0
    r22 = q1.norm(1e-6)
    q1/=r22
    Q = ti.Matrix.cols([q0,q1])
    R = ti.Matrix([[r11,r12],[0,r22]])
    return Q,R

@ti.func
def particle_parameters_on_grid(pos):
    base = (pos * inv_dx - 0.5).cast(int) #每个顶点所在网格的左下角格点坐标
    fx = pos * inv_dx - base.cast(float) #每个格点相对左下角位置
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
    return base, fx, w

@ti.kernel
def Particle_To_Grid():
    for p in x2:
        base, fx, w = particle_parameters_on_grid(x2[p])
        affine = C2[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])
            weight = w[i][0]*w[j][1]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass * (v2[p] +  affine@dpos)


    for p in x3:
        base, fx, w = particle_parameters_on_grid(x3[p])
        affine = C3[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])
            weight = w[i][0]*w[j][1]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass * (v3[p] +  affine@dpos)

# 更新grid_force的时候有用到
@ti.kernel
def Grid_Force():
    for p in x3:
        l, n = GetType2FromType3(p)
        base, fx, w = particle_parameters_on_grid(x3[p])

        dw_dx_d = ti.Matrix.rows([fx-1.5, 2*(1.0-fx), fx-0.5]) * inv_dx
        base_l, fx_l, w_l = particle_parameters_on_grid(x2[l])
        base_n, fx_n, w_n = particle_parameters_on_grid(x2[n])

        Q, R = QR2(F3[p])
        r11, r12, r22 = R[0,0], R[0,1], R[1,1] 

        K = ti.Matrix.rows([
            [E*r11*(r11-1)+gamma*r12**3, gamma * r12**2 * r22],
            [gamma * r12**2 * r22,  -k * (1 - r22)**2 * r22 * float(r22 <= 1)]
            ])
        dphi_dF = Q@ K @ R.inverse().transpose()# Q.inverse().transpose() = Q.transpose().transpose() = Q

        dp_c1 = ti.Vector([D3[p][0,1],D3[p][1,1]])

        dphi_dF_c1 = ti.Vector([dphi_dF[0,1],dphi_dF[1,1]])

        Dp_inv_c0 = ti.Vector([D3_inv[p][0,0],D3_inv[p][1,0]])

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])

            # technical document .(15) part 1
            weight_l = w_l[i][0] * w_l[j][1]
            weight_n = w_n[i][0] * w_n[j][1]

            f_2 = dphi_dF @ Dp_inv_c0
            grid_f[base_l + offset]+=  volume2 * weight_l * f_2
            grid_f[base_n + offset]+= -volume2 * weight_n * f_2

            # dphi w / x
            dw_dx = ti.Vector([ dw_dx_d[i, 0] * w[j][1], w[i][0] * dw_dx_d[j, 1] ])
            # technical document .(15) part 2
            grid_f[base + offset] += -volume3 * dphi_dF_c1* dw_dx.dot( dp_c1 )
    
    # spring force, bending parameter
    for p in range((ln_type2-2) * N_Line):
        nl = p // (ln_type2-2)
        
        v0 = p + nl* 2
        v1 = v0 + 2

        base_0, fx_0, w_0 = particle_parameters_on_grid(x2[v0])
        base_1, fx_1, w_1 = particle_parameters_on_grid(x2[v1])        
        dir_x = x2[v1] - x2[v0]
        dist = dir_x.norm(1e-9)
        dir_x /= dist
        fn = dist- 2.0 * length_type2
        f = -1000 * fn * dir_x

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])

            weight_0 = w_0[i][0] * w_0[j][1]
            weight_1 = w_1[i][0] * w_1[j][1]
            
            grid_f[base_0 + offset] -= weight_0 * f
            grid_f[base_1 + offset] += weight_1 * f

    

@ti.kernel
def Grid_Collision():

    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] +=  grid_f[I] * dt
            grid_v[I] /= grid_m[I]
            grid_v[I].y -= dt * 9.80

            #circle collision
            dist = I*dx - Circle_Center
            if dist.x**2 + dist.y**2 < Circle_Radius* Circle_Radius :
                dist = dist.normalized()
                grid_v[I] -= dist * min(0, grid_v[I].dot(dist) )
                grid_v[I] *= 0.9  #circle friction

            # flip 
            condition = (I<bound) & (grid_v[I]<0) | (I>n_grid-bound) & (grid_v[I]>0)
            grid_v[I] = 0 if condition else grid_v[I]
 

@ti.kernel
def Grid_To_Particle():
    for p in x2:
        base, fx, w = particle_parameters_on_grid(x2[p])

        new_v = ti.math.vec2(0.0)
        new_C = ti.math.mat2(0.0)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        v2[p] = new_v
        x2[p] += dt * v2[p]
        C2[p] = new_C

    for p in x3:
        base, fx, w = particle_parameters_on_grid(x3[p])
        new_C = ti.math.mat2(0.0)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        C3[p] = new_C


@ti.kernel
def Update_Particle_State():
    for p in x3:
        l, n = GetType2FromType3(p)
        v3[p] = 0.5 * (v2[l] + v2[n])
        x3[p] = 0.5 * (x2[l] + x2[n])

        dp0 = x2[n] - x2[l]
        dp1 = ti.Vector([D3[p][0,1],D3[p][1,1]])
        dp1 += dt * C3[p] @ dp1
        D3[p] = ti.Matrix.cols([dp0,dp1])
        F3[p] = D3[p] @ D3_inv[p]


cf = 0.05
@ti.kernel
def Return_Mapping():
    for p in x3:
        Q,R = QR2(F3[p])
        r12 = R[0,1]
        r22 = R[1,1]

        # 这个mapping好像也有问题？
        #cf = 0
        if r22 < 0:
            r12 = 0
            #r22 = max(r22, -1)
            r22 = 0
        elif r22 > 1:
            r12 = 0
            r22 = 1
        else:
            f = (gamma/k)*ti.abs(r12) - cf*((1.0-r22)**2)
            if f > 0:
                scale = (k/gamma)*cf*((1.0-r22)**2)/ti.abs(r12)
                r12*= scale


        R[0,1] = r12
        R[1,1] = r22

        F3[p] = Q@R
        D3[p] = F3[p]@D3_inv[p].inverse()

@ti.kernel
def Reset():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
        grid_f[i, j] = [0.0,0.0]


#get type2 from type3
@ti.func
def GetType2FromType3(index):
    index += index // ln_type3
    return index, index+1

@ti.kernel
def initialize():
    for i in range(n_type2):
        ln_num = i // ln_type2
        # 至少修改成竖着的 为什么变成竖的一下就炸了？没有办法 为什么会炸？
        # 可能是算法里面有写死的成分
        x2[i] = ti.Vector([start_pos[0] + (i - ln_num * ln_type2) * length_type2, 
                           start_pos[1] + ln_num * line_interval])
        #x2[i] = ti.Vector([start_pos[0] + ln_num * line_interval, 
        #                   start_pos[1] - (i - ln_num * ln_type2) * length_type2])

        v2[i] = ti.Matrix([0, 0])
        C2[i] =  ti.Matrix([[0,0],[0,0]])

    for i in range(n_segment):
        l, n = GetType2FromType3(i)

        x3[i] = 0.5*(x2[l] + x2[n])
        v3[i] = ti.Matrix([0, 0])
        F3[i] = ti.Matrix([[1.0, 0.0],[0.0, 1.0] ])
        C3[i] =  ti.Matrix([[0,0],[0,0]])   

        dp0 = x2[n] - x2[l]
        dp1 = (ROT90@dp0).normalized(1e-6)


        D3[i] = ti.Matrix.cols([dp0,dp1])
        D3_inv[i] = D3[i].inverse()     

        
def main():
    initialize()

    Circle_Center_field = ti.Vector.field(2, dtype = float, shape = 1)
    Circle_Center_field[0] = Circle_Center

    window = ti.ui.Window("MPM_Cloth", (800, 800))
    canvas = window.get_canvas()
    while window.running:
        for _ in range(50):
            Reset()
            Particle_To_Grid()
            Grid_Force()
            Grid_Collision()
            Grid_To_Particle()
            Update_Particle_State()
            Return_Mapping()

        canvas.set_background_color((0.067, 0.184, 0.255))
        
        canvas.circles(Circle_Center_field, radius=Circle_Radius, color=(0, 0.5, 0.5))

        # 为什么circle可以这么精细？
        canvas.circles(x2, radius=Circle_Radius * 0.008, color=(0.8, 0.5, 0.5))

        window.show()

if __name__ == "__main__":
    main()