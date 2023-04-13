
import taichi as ti
import math

ti.init(arch=ti.cpu)

# global control
paused = True
damping_toggle = ti.field(ti.i32, ())
curser = ti.Vector.field(2, ti.f32, ())
picking = ti.field(ti.i32,())
using_auto_diff = False

# procedurally setting up the cantilever
init_x, init_y = 0.1, 0.4
N_x = 15
N_y = 8
# N_x = 2
# N_y = 2
N = N_x*N_y
N_edges = (N_x-1)*N_y + N_x*(N_y - 1) + (N_x-1) * (N_y-1)

N_triangles = 2 * (N_x-1) * (N_y-1)
dx = 1/32
curser_radius = dx/2

# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())

# time-step size (for simulation, 16.7ms)
h = 16.7e-4
# substepping
substepping = 1000
# time-step size (for time integration)
dt = h/substepping

# simulation components
x = ti.Vector.field(2, ti.f32, N, needs_grad=True)
v = ti.Vector.field(2, ti.f32, N)
f = ti.Vector.field(2, ti.f32, N)
B = ti.Matrix.field(2, 2, ti.f32, N_triangles)
W = ti.field(ti.f32, N_triangles)

# geometric components
triangles = ti.Vector.field(3, ti.i32, N_triangles)
edges = ti.Vector.field(2, ti.i32, N_edges)
c2e = ti.Vector.field(3, ti.i32, N_triangles)


def ij_2_index(i, j): return i * N_y + j


# -----------------------meshing and init----------------------------
@ti.kernel
def meshing():
    # setting up triangles
    for i,j in ti.ndrange(N_x - 1, N_y - 1):
        # triangle id
        tid = (i * (N_y - 1) + j) * 2
        triangles[tid][0] = ij_2_index(i, j)
        triangles[tid][1] = ij_2_index(i + 1, j)
        triangles[tid][2] = ij_2_index(i, j + 1)

        tid = (i * (N_y - 1) + j) * 2 + 1
        triangles[tid][0] = ij_2_index(i, j + 1)
        triangles[tid][1] = ij_2_index(i + 1, j + 1)
        triangles[tid][2] = ij_2_index(i + 1, j)

    # setting up edges
    # edge id
    eid_base = 0

    # horizontal edges
    for i in range(N_x-1):
        for j in range(N_y):
            eid = eid_base+i*N_y+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i+1, j)]

    eid_base += (N_x-1)*N_y
    # vertical edges
    for i in range(N_x):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i, j+1)]

    eid_base += N_x*(N_y-1)
    # diagonal edges
    for i in range(N_x-1):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i, j+1), ij_2_index(i+1, j)]

@ti.kernel
def find_c2e():
    for c in range(N_triangles):
        u,v,w = triangles[c][0], triangles[c][1], triangles[c][2]
        e1,e2,e3=ti.Vector([u,v]),ti.Vector([u,w]),ti.Vector([w,v])
        index = 0
        for e in range(N_edges):           
            if all(e1 == edges[e]):
                c2e[c][0] = index
            elif all(e2 == edges[e]):
                c2e[c][1] = index
            elif all(e3 == edges[e]):
                c2e[c][2] = index
            index += 1



@ti.kernel
def initialize():
    YoungsModulus[None] = 1e6
    PoissonsRatio[None] = 0.3
    damping_toggle[None] = True

    paused = True
    # init position and velocity
    for i, j in ti.ndrange(N_x, N_y):
        index = ij_2_index(i, j)
        x[index] = ti.Vector([init_x + i * dx, init_y + j * dx])
        v[index] = ti.Vector([0.0, 0.0])


@ti.func
def compute_D(i):
    a = triangles[i][0]
    b = triangles[i][1]
    c = triangles[i][2]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a]])

@ti.kernel
def initialize_elements():
    for i in range(N_triangles):
        Dm = compute_D(i)
        B[i] = Dm.inverse()
        W[i] = ti.abs(Dm.determinant())/2

# ----------------------core-----------------------------
@ti.func
def compute_R_2D(F):
    R, S = ti.polar_decompose(F, ti.f32)
    return R

@ti.kernel
def compute_gradient():
    # clear gradient
    for i in f:
        f[i] = ti.Vector([0, 0])
    # gradient of elastic potential
    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds@B[i]
        # co-rotated linear elasticity
        R = compute_R_2D(F)
        Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
        # first Piola-Kirchhoff tensor
        P = 2*LameMu[None]*(F-R) + LameLa[None]*((R.transpose())@F-Eye).trace()*R
        #assemble to gradient
        H = W[i] * P @ (B[i].transpose())

        a,b,c = triangles[i][0],triangles[i][1],triangles[i][2]
        gb = ti.Vector([H[0,0], H[1, 0]])
        gc = ti.Vector([H[0,1], H[1, 1]])
        ga = -gb-gc
        f[a] += ga
        f[b] += gb
        f[c] += gc     


@ti.kernel
def compute_gradient_stVK():
    for i in range(N_edges):
        f[i] = ti.Vector([0, 0])
    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds @ B[i]
        # stVK
        Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
        stvk=0.5*((F.transpose() @ F)-Eye)
        # first Piola-Kirchhoff tensor
        #P= F@(2*LameMu[None]* 0.5*((F.transpose() @ F)-Eye) +LameLa[None]*stvk.trace()*Eye)
        P= F@(2*LameMu[None]* stvk +LameLa[None]*stvk.trace()*Eye)
        # assemble to gradient
        H = W[i] * P @ (B[i].transpose())
        a, b, c = triangles[i][0], triangles[i][1], triangles[i][2]
        gb = ti.Vector([H[0, 0], H[1, 0]])
        gc = ti.Vector([H[0, 1], H[1, 1]])
        ga = -gb - gc
        f[a] += ga
        f[b] += gb
        f[c] += gc

@ti.kernel
def compute_gradient_NeoHookean():
    for i in range(N_edges):
        f[i] = ti.Vector([0, 0])
    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds @ B[i]

        ## NeoHookean
        F_inv_T = F.inverse().transpose()
        J = max(F.determinant(), 0.01)
        P = LameMu[None] * (F - F_inv_T) + LameLa[None] * ti.log(J) * F_inv_T

        # assemble to gradient
        H = W[i] * P @ (B[i].transpose())
        a, b, c = triangles[i][0], triangles[i][1], triangles[i][2]
        gb = ti.Vector([H[0, 0], H[1, 0]])
        gc = ti.Vector([H[0, 1], H[1, 1]])
        ga = -gb - gc
        f[a] += ga
        f[b] += gb
        f[c] += gc



#====================================================================================================

HessianEdge = ti.Vector.field(2, dtype=ti.f32, shape=N_edges) 
HessianVert = ti.Vector.field(2, dtype=ti.f32, shape=N_triangles) 

b = ti.Vector.ndarray(2, dtype=ti.f32, shape=N_triangles)
r0 = ti.Vector.ndarray(2, dtype=ti.f32, shape=N_triangles)
z0 = ti.Vector.ndarray(2, dtype=ti.f32, shape=N_triangles)
p0 = ti.Vector.ndarray(2, dtype=ti.f32, shape=N_triangles)
alpha_scalar = ti.ndarray(ti.f32, shape=())
beta_scalar = ti.ndarray(ti.f32, shape=())

dot_ans = ti.field(ti.f32, shape=())
r_2_scalar = ti.field(ti.f32, shape=())


@ti.kernel
def get_matrix(c2e: ti.types.template(), triangles: ti.types.template()):

    for c in range(N_triangles):

        verts = triangles[c]
        W_c = W[c]
        B_c = B[c]

        hes = ti.Matrix.zero(ti.f32, 6, 6)

        for u in ti.static(range(3)):
            for d in ti.static(range(2)):

                dD = ti.Matrix.zero(ti.f32, 2, 2)
                if ti.static(u == 2):
                    for j in ti.static(range(2)):
                        dD[d, j] = -1
                else:
                    dD[d, u] = 1

                dF = dD @ B_c
                dP = 2.0 * LameMu[None] * dF
                dH = -W_c * dP @ B_c.transpose() #3*3

                for i in ti.static(range(2)):
                    for j in ti.static(range(2)):
                        hes[i * 2 + j, u * 2 + d] = -dt**2 * dH[j, i] # 0-4 row
                        hes[2 * 2 + j, u * 2 + d] += dt**2 * dH[j, i] # 5/6 row

        z = 0
        for u_i in ti.static(range(3)):
            u = verts[u_i]
            for v_i in ti.static(range(3)):
                v = verts[v_i]
                if u < v:
                    HessianEdge[c2e[c][z]][0] += hes[u_i*3, v_i*3]
                    HessianEdge[c2e[c][z]][1] += hes[u_i*3+1, v_i*3+1]
                    z += 1

        for zz in ti.static(range(3)):
            HessianVert[verts[zz]][0] += hes[zz*2, zz*2]
            HessianVert[verts[zz]][1] += hes[zz*2+1, zz*2+1]



#HessianVert就是对角线元素
@ti.kernel
def matmul_edge(ret: ti.types.ndarray(), vel: ti.types.ndarray(), edges: ti.types.ndarray()):

    for i in ret:
        # HessianVert = -dt**2 * dH, Avn = (m-dt**2*dH)*v
        ret[i][0] = HessianVert[i][0] * vel[i][0]
        ret[i][1] = HessianVert[i][1] * vel[i][1]
        ret[i][2] = HessianVert[i][2] * vel[i][2]
        ret[i] += m[i] * vel[i]

    for e in edges:
        u, v = edges[e][0], edges[e][1]
        for j in ti.static(range(3)):
            ret[u][j] += HessianEdge[e][j] * vel[v][j]
            ret[v][j] += HessianEdge[e][j] * vel[u][j]


#M是preconditioner z0=D^-1*r0
@ti.kernel
def Jacobi_pre(res:ti.types.ndarray(), r0:ti.types.ndarray()):
    for i in range(n_verts):
        res[i][0] = r0[i][0] / (m[i] + HessianVert[i][0])
        res[i][1] = r0[i][1] / (m[i] + HessianVert[i][1])
        res[i][2] = r0[i][2] / (m[i] + HessianVert[i][2])


#CG parts
@ti.kernel
def add(ans: ti.types.ndarray(), a: ti.types.ndarray(), k: ti.f32,
        b: ti.types.ndarray()):
    for i in ans:
        ans[i] = a[i] + k * b[i]


@ti.kernel
def add_scalar_ndarray(ans: ti.types.ndarray(), a: ti.types.ndarray(),
                       k: ti.f32, scalar: ti.types.ndarray(),
                       b: ti.types.ndarray()):
    for i in ans:
        ans[i] = a[i] + k * scalar[None] * b[i]


@ti.kernel
def dot2scalar(a: ti.types.ndarray(), b: ti.types.ndarray()):
    dot_ans[None] = 0.0
    for i in a:
        dot_ans[None] += a[i].dot(b[i])

@ti.kernel
def get_b(v: ti.types.ndarray(), b: ti.types.ndarray(), f: ti.types.ndarray()):
    for i in b:
        # M*v + h*f
        b[i] = m[i] * v[i] + dt * f[i]


@ti.kernel
def ndarray_to_ndarray(ndarray: ti.types.ndarray(), other: ti.types.ndarray()):
    for I in ti.grouped(ndarray):
        ndarray[I] = other[I]


@ti.kernel
def fill_ndarray(ndarray: ti.types.ndarray(), val: ti.f32):
    for I in ti.grouped(ndarray):
        ndarray[I] = [val, val, val]


@ti.kernel
def init_r_2():
    r_2_scalar[None] = dot_ans[None]


@ti.kernel
def update_alpha(alpha_scalar: ti.types.ndarray(), numerator: ti.template(), denominator: ti.template()):
    alpha_scalar[None] = numerator[None] / (denominator[None] + epsilon)


@ti.kernel
def update_beta_r_2(beta_scalar: ti.types.ndarray(), numerator: ti.template(), denominator: ti.template()):
    beta_scalar[None] = numerator[None] / (denominator[None] + epsilon)
    denominator[None] = numerator[None]


def Jacobi_pcg(it):

    get_force(x, ox, f, vertices, gravity[0], gravity[1], gravity[2])

    #print("=======",it,"=======", f[0],f[100],f[300])

    get_b(v, b, f)
    matmul_edge(mul_ans, v, edges) #mul_ans = A*v0    
    add(r0, b, -1, mul_ans) # r0 = b-Ax0 = b-mul_ans

    Jacobi_pre(z0, r0)

    ndarray_to_ndarray(p0, z0) # p0 = z0
    dot2scalar(r0, z0) # dot_ans = r0*z0
    init_r_2() # r_2_scalar = dot_ans
    
    CG_ITERS = 10
    #iter_error = 10
    while CG_ITERS > 0:
    #while iter_error > 1:
        
        matmul_edge(mul_ans, p0, edges) #mul_ans = A*p0
        dot2scalar(p0, mul_ans) #dot_ans = p0*A*p0
        update_alpha(alpha_scalar, r_2_scalar, dot_ans) #alpha=r_2_scalar/dot_ans
        add_scalar_ndarray(v, v, 1, alpha_scalar, p0) #vn = vn+alpha*p0
        add_scalar_ndarray(r0, r0, -1, alpha_scalar, mul_ans) #r0 = r0-alpha*A*p0

        Jacobi_pre(z0, r0) #z0=M^-1*r0

        dot2scalar(r0, z0) #dot_ans = r0*z0
        update_beta_r_2(beta_scalar, dot_ans, r_2_scalar) #beta=dot_ans/r_2_sca, r_2_sca=dot_ans
        add_scalar_ndarray(p0, z0, 1, beta_scalar, p0)

        CG_ITERS -= 1
        #iter_error = dot_ans[None]
        #print("iter", 10-CG_ITERS, "Residual", iter_error)
        
    fill_ndarray(f, 0)
    add(x, x, dt, v)

#=======================================================================================





@ti.kernel
def update():
    # perform time integration
    for i in range(N):
        # symplectic integration
        # elastic force + gravitation force, divding mass to get the acceleration
        acc = -f[i]/m - ti.Vector([0.0, g])
        v[i] += dt*acc
        x[i] += dt*v[i]

    # explicit damping (ether drag)
    for i in v:
        if damping_toggle[None]:
            v[i] *= ti.exp(-dt*5)

    # enforce boundary condition
    # drag one point
    for i in range(N):
        if picking[None]:           
            r = x[i]-curser[None]
            if r.norm() < curser_radius:
                x[i] = curser[None]
                v[i] = ti.Vector([0.0, 0.0])
                pass

    #drag a line
    #for i in range(N):
    #    if picking[None]:           
    #        r = x[i].x-curser[None].x
    #        if ti.abs(r) < curser_radius:
    #            x[i].x = curser[None].x
    #            v[i] = ti.Vector([0.0, 0.0])
    #            pass

    for j in range(N_y):
        ind = ij_2_index(0, j)
        v[ind] = ti.Vector([0, 0])
        x[ind] = ti.Vector([init_x, init_y + j * dx])  # rest pose attached to the wall

    for i in range(N):
        if x[i][0] < init_x:
            x[i][0] = init_x
            v[i][0] = 0


@ti.kernel
def updateLameCoeff():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))

# init once and for all
meshing()
find_c2e()
initialize()
initialize_elements()
updateLameCoeff()

#print(c2e)

gui = ti.GUI('Linear FEM', (700, 700))
while gui.running:

    picking[None]=0

    # key events
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'r':
            initialize()
        elif e.key == ti.GUI.SPACE:
            paused = not paused
        elif e.key =='d' or e.key == 'D':
            damping_toggle[None] = not damping_toggle[None]
        
        updateLameCoeff()

    if gui.is_pressed(ti.GUI.LMB):
        curser[None] = gui.get_cursor_pos()
        picking[None] = 1

    # numerical time integration
    if not paused:
        for i in range(substepping):
            compute_gradient_stVK()
            #compute_gradient()
            #compute_gradient_NeoHookean()
            update()

    # render
    pos = x.to_numpy()
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        gui.line((pos[a][0], pos[a][1]),
                 (pos[b][0], pos[b][1]),
                 radius=1,
                 color=0xFFFF00)
    gui.line((init_x, 0.0), (init_x, 1.0), color=0xFFFFFF, radius=4)

    if picking[None]:
        gui.circle((curser[None][0], curser[None][1]), radius=curser_radius*800, color=0xFF8888)

    gui.show()