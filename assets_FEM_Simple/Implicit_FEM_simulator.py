
import argparse
import os
import numpy as np
import taichi as ti
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=3)
args = parser.parse_args()

ti.init(arch=ti.cuda)
# ti.init(arch=ti.cpu, debug=False)

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def get_rel_path(*segs):
    return os.path.join(SCRIPT_PATH, *segs)

c2e_np = np.load(get_rel_path('ArmadilloModel/c2e.npy'))# cell to edge
vertices_np = np.load(get_rel_path('ArmadilloModel/vertices_np.npy'))
indices_np = np.load(get_rel_path('ArmadilloModel/indices_np.npy'))
edges_np = np.load(get_rel_path('ArmadilloModel/edges_np.npy'))
ox_np = np.load(get_rel_path('ArmadilloModel/ox_np.npy'))

n_edges = edges_np.shape[0]
n_verts = ox_np.shape[0]
n_cells = c2e_np.shape[0]
n_faces = indices_np.shape[0]

E, nu = 5e5, 0.0
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # lambda = 0
density = 1000.0
epsilon = 1e-6
num_substeps = 2
dt = 1.0/24/num_substeps

x = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
v = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
f = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
#conjugate gradient
mul_ans = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
#conjugate residual 
cr_mul_ans = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)

m = ti.field(dtype=ti.f32, shape=n_verts)
B = ti.Matrix.field(args.dim, args.dim, dtype=ti.f32, shape=n_cells)
W = ti.field(dtype=ti.f32, shape=n_cells)

gravity = [0, -9.8, 0]

ox = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
vertices = ti.Vector.ndarray(4, dtype=ti.i32, shape=n_cells)
indices = ti.field(ti.i32, shape=n_faces * 3)
edges = ti.Vector.ndarray(2, dtype=ti.i32, shape=n_edges) #2954
c2e = ti.Vector.ndarray(6, dtype=ti.i32, shape=n_cells)

HessianEdge = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_edges) #2954
HessianVert = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts) #616

b = ti.Vector.ndarray(3, dtype=ti.f32, shape=n_verts)
r0 = ti.Vector.ndarray(3, dtype=ti.f32, shape=n_verts)
z0 = ti.Vector.ndarray(3, dtype=ti.f32, shape=n_verts)
p0 = ti.Vector.ndarray(3, dtype=ti.f32, shape=n_verts)
alpha_scalar = ti.ndarray(ti.f32, shape=())
beta_scalar = ti.ndarray(ti.f32, shape=())

dot_ans = ti.field(ti.f32, shape=())
r_2_scalar = ti.field(ti.f32, shape=())

ox.from_numpy(ox_np)
vertices.from_numpy(vertices_np)
indices.from_numpy(indices_np.reshape(-1))
edges.from_numpy(np.array(list(edges_np)))
c2e.from_numpy(c2e_np)


@ti.kernel
def clear_field():
    for I in ti.grouped(HessianVert):
        HessianVert[I].fill(0)
    for I in ti.grouped(HessianEdge):
        HessianEdge[I].fill(0)


@ti.func
def Ds(verts, x: ti.template()):
    col1 = x[verts[0]] - x[verts[3]] + epsilon
    col2 = x[verts[1]] - x[verts[3]] + epsilon
    col3 = x[verts[2]] - x[verts[3]] + epsilon 
    return ti.Matrix.cols([col1, col2, col3])


@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V


@ti.func
def get_force_func(c, verts, x: ti.template(), f: ti.template()):
    F = Ds(verts, x) @ B[c]
    P = ti.Matrix.zero(ti.f32, 3, 3)
    U, sig, V = ssvd(F)
    P = 2 * mu * (F - U @ V.transpose())
    H = -W[c] * P @ B[c].transpose()
    for i in ti.static(range(3)):
        force = ti.Vector([H[j, i] for j in range(3)])
        f[verts[i]] += force
        f[verts[3]] -= force


@ti.kernel
def get_force(x: ti.types.ndarray(), f: ti.types.ndarray(),
        vertices: ti.types.ndarray(), g_x: ti.f32, g_y: ti.f32, g_z: ti.f32):
    for c in vertices:
        get_force_func(c, vertices[c], x, f)
    for u in f:
        g = ti.Vector([g_x, g_y, g_z])
        f[u] += g * m[u]

# 不对啊？？我搞不明白 为什么get_matrix只更新了一次？？？
@ti.kernel
def get_matrix(c2e: ti.types.ndarray(), vertices: ti.types.ndarray()):
    for c in vertices:
        verts = vertices[c]
        W_c = W[c]
        B_c = B[c]
        hes = ti.Matrix.zero(ti.f32, 12, 12)

        for u in ti.static(range(4)):
            for d in ti.static(range(3)):
                dD = ti.Matrix.zero(ti.f32, 3, 3)
                if ti.static(u == 3):
                    for j in ti.static(range(3)):
                        dD[d, j] = -1
                else:
                    dD[d, u] = 1
                dF = dD @ B_c
                dP = 2.0 * mu * dF
                dH = -W_c * dP @ B_c.transpose() #3*3
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        hes[i * 3 + j, u * 3 + d] = -dt**2 * dH[j, i] # 0-8 row
                        hes[3 * 3 + j, u * 3 + d] += dt**2 * dH[j, i] # 9/10/11 row

        z = 0
        for u_i in ti.static(range(4)):
            u = verts[u_i]
            for v_i in ti.static(range(4)):
                v = verts[v_i]
                if u < v:
                    HessianEdge[c2e[c][z]][0] += hes[u_i*3, v_i*3]
                    HessianEdge[c2e[c][z]][1] += hes[u_i*3+1, v_i*3+1]
                    HessianEdge[c2e[c][z]][2] += hes[u_i*3+2, v_i*3+2]
                    z += 1

        for zz in ti.static(range(4)):
            HessianVert[verts[zz]][0] += hes[zz*3, zz*3]
            HessianVert[verts[zz]][1] += hes[zz*3+1, zz*3+1]
            HessianVert[verts[zz]][2] += hes[zz*3+2, zz*3+2]


@ti.kernel
def matmul_edge(ret: ti.types.ndarray(), vel: ti.types.ndarray(), edges: ti.types.ndarray()):

    for i in ret:
        # HessianVert = -h^2*dH, Avn = (m-h^2*dH)*v
        ret[i][0] = HessianVert[i][0] * vel[i][0]
        ret[i][1] = HessianVert[i][1] * vel[i][1]
        ret[i][2] = HessianVert[i][2] * vel[i][2]
        ret[i] += m[i] * vel[i]

    for e in edges:
        u, v = edges[e][0], edges[e][1]
        for j in ti.static(range(3)):
            ret[u][j] += HessianEdge[e][j] * vel[v][j]
            ret[v][j] += HessianEdge[e][j] * vel[u][j]


@ti.kernel
def Jacobi_pre(res:ti.types.ndarray(), r0:ti.types.ndarray()):
    for i in range(n_verts):
        res[i][0] = r0[i][0] / (m[i] + HessianVert[i][0])
        res[i][1] = r0[i][1] / (m[i] + HessianVert[i][1])
        res[i][2] = r0[i][2] / (m[i] + HessianVert[i][2])


@ti.kernel
def Cholesky_pre(M:ti.types.template(), res:ti.types.ndarray()):
    pass


n_mg_levels = 5 
pre_and_post_smoothing = 10
bottom_smoothing = 50
mg_r0 = ti.Matrix.ndarray(n_verts, 3, dtype=ti.f32, shape=n_mg_levels) 
mg_z0 = ti.Matrix.ndarray(n_verts, 3, dtype=ti.f32, shape=n_mg_levels)
mg_temp_r0 = ti.field(dtype=ti.f32, shape=(n_verts, 3))
mg_temp_z0 = ti.field(dtype=ti.f32, shape=(n_verts, 3))
mg_temp_LUz0 = ti.field(dtype=ti.f32, shape=(n_verts, 3))

@ti.kernel
def MG_restrict(l: ti.template(), mg_z0:ti.types.ndarray(), mg_r0:ti.types.ndarray(), edges: ti.types.ndarray()):

    for i,j in ti.ndrange(n_verts // 2**l, 3):
        mg_temp_z0[i,j] = 0.0

    # A = (m + Hessian)
    for i in range(n_verts // 2**l):
        for j in range(2**l):
            mg_temp_z0[i,0] += 0.5**l * ( m[(2**l)*i+j] + HessianVert[(2**l)*i+j][0] ) * mg_z0[l][i,0]
            mg_temp_z0[i,1] += 0.5**l * ( m[(2**l)*i+j] + HessianVert[(2**l)*i+j][1] ) * mg_z0[l][i,1]
            mg_temp_z0[i,2] += 0.5**l * ( m[(2**l)*i+j] + HessianVert[(2**l)*i+j][2] ) * mg_z0[l][i,2]

    for e in edges:
        u, v = edges[e][0]//2**l, edges[e][1]//2**l
        for j in ti.static(range(3)):
            mg_temp_z0[u,j] += 0.5**l * HessianEdge[e][j] * mg_z0[l][v,j]
            mg_temp_z0[v,j] += 0.5**l * HessianEdge[e][j] * mg_z0[l][u,j]

    for i,j in ti.ndrange(n_verts // 2**l, 3):
        res = mg_r0[l][i,j] - mg_temp_z0[i,j]
        mg_r0[l+1][i//2,j] += 0.5 * res

@ti.kernel
def MG_smooth(l: ti.template(), phase: ti.template(), mg_z0:ti.types.ndarray(), mg_r0:ti.types.ndarray(), edges: ti.types.ndarray()):

    for i,j in ti.ndrange(n_verts, 3):
        mg_temp_z0[i,j] = 0
        mg_temp_r0[i,j] = 0
        mg_temp_LUz0[i,j] = 0

    for i in range(n_verts // 2**l):
        for j in range(2**l):
            # 0.5*(r/d)
            mg_temp_r0[i,0] += 0.5**l * (mg_r0[l][i,0] / (m[(2**l)*i+j] + HessianVert[(2**l)*i+j][0]))
            mg_temp_r0[i,1] += 0.5**l * (mg_r0[l][i,1] / (m[(2**l)*i+j] + HessianVert[(2**l)*i+j][1]))
            mg_temp_r0[i,2] += 0.5**l * (mg_r0[l][i,2] / (m[(2**l)*i+j] + HessianVert[(2**l)*i+j][2]))

    for e in edges:
        u, v = edges[e][0]//2**l, edges[e][1]//2**l
        for j in ti.static(range(3)):
            mg_temp_LUz0[u,j] = HessianEdge[e][j] * mg_z0[l][v,j]
            mg_temp_LUz0[v,j] = HessianEdge[e][j] * mg_z0[l][u,j]

    for i in range(n_verts // 2**l):
        for j in range(2**l):
            mg_temp_z0[i,0] += 0.5**l * (mg_temp_LUz0[i,0] / (m[(2**l)*i+j] + HessianVert[(2**l)*i+j][0]))
            mg_temp_z0[i,1] += 0.5**l * (mg_temp_LUz0[i,1] / (m[(2**l)*i+j] + HessianVert[(2**l)*i+j][1]))
            mg_temp_z0[i,2] += 0.5**l * (mg_temp_LUz0[i,2] / (m[(2**l)*i+j] + HessianVert[(2**l)*i+j][2]))

    for i,j in ti.ndrange(n_verts // 2**l, 3):
        mg_z0[l][i,j] = mg_temp_r0[i,j] - mg_temp_z0[i,j]


@ti.kernel
def MG_prolongate(l: ti.template(), mg_z0:ti.types.ndarray()):
    for i,j in ti.ndrange(n_verts // 2**l, 3):
        mg_z0[l][i,j] = mg_z0[l+1][i//2, j]


@ti.kernel
def init_mg_vector(l:ti.types.template(), mgvector:ti.types.ndarray()):
    for i,j in ti.ndrange(n_verts// 2**l, 3):
        mgvector[l][i,j] = 0


@ti.kernel
def r02mgvector(mg_r0:ti.types.ndarray(), r0:ti.types.ndarray()):
    for i,j in ti.ndrange(n_verts, 3):
        mg_r0[0][i,j] = r0[i][j]

        
@ti.kernel
def mgvector2r0(r0:ti.types.ndarray(), mg_r0:ti.types.ndarray()):
    for i,j in ti.ndrange(n_verts, 3):
        r0[i][j] = mg_r0[0][i,j]

def Multigrid_pre():
   
    init_mg_vector(0, mg_z0)

    for l in range(n_mg_levels-1):
        for i in range(pre_and_post_smoothing << l):
            MG_smooth(l, 0, mg_z0, mg_r0, edges)
            MG_smooth(l, 1, mg_z0, mg_r0, edges)

        init_mg_vector(l+1, mg_z0)
        init_mg_vector(l+1, mg_r0)
        MG_restrict(l, mg_z0, mg_r0, edges)

    for i in range(bottom_smoothing):
        MG_smooth(n_mg_levels-1, 0, mg_z0, mg_r0, edges)
        MG_smooth(n_mg_levels-1, 1, mg_z0, mg_r0, edges)

    for l in reversed(range(n_mg_levels-1)):
        MG_prolongate(l, mg_z0)
        for i in range(pre_and_post_smoothing << l):
            MG_smooth(l, 1, mg_z0, mg_r0, edges)
            MG_smooth(l, 0, mg_z0, mg_r0, edges)


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



res_scalar = ti.field(ti.f32, ())
@ti.kernel
def itererror(a: ti.types.ndarray()):
    res_scalar[None] = 0.0
    for I in ti.grouped(a):
        res_scalar[None] += a[I].dot(a[I])
    res_scalar[None] = ti.sqrt(res_scalar[None])


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



def MG_pcg(it):
    
    get_force(x, f, vertices, gravity[0], gravity[1], gravity[2])
    get_b(v, b, f)
    matmul_edge(mul_ans, v, edges) #mul_ans = A*v0    
    add(r0, b, -1, mul_ans) # r0 = b-Ax0 = b-mul_ans
    r02mgvector(mg_r0, r0)
    Multigrid_pre()
    mgvector2r0(z0, mg_z0)
    ndarray_to_ndarray(p0, z0) # p0 = z0
    dot2scalar(r0, z0) # dot_ans = r0*z0
    init_r_2() # r_2_scalar = dot_ans
    
    for CG_ITERS in range(30):
        
        matmul_edge(mul_ans, p0, edges) #mul_ans = A*p0
        dot2scalar(p0, mul_ans) #dot_ans = p0*A*p0
        update_alpha(alpha_scalar, r_2_scalar, dot_ans) #alpha=r_2_scalar/dot_ans
        add_scalar_ndarray(v, v, 1, alpha_scalar, p0) #vn = vn+alpha*p0
        add_scalar_ndarray(r0, r0, -1, alpha_scalar, mul_ans) #r0 = r0-alpha*A*p0
        r02mgvector(mg_r0, r0)
        Multigrid_pre()
        mgvector2r0(z0, mg_z0)
        dot2scalar(r0, z0) #dot_ans = r0*z0
        update_beta_r_2(beta_scalar, dot_ans, r_2_scalar) #beta=dot_ans/r_2_sca, r_2_sca=dot_ans
        add_scalar_ndarray(p0, z0, 1, beta_scalar, p0)
        
    fill_ndarray(f, 0)
    add(x, x, dt, v)

def MG_pcr(it):
    
    get_force(x, f, vertices, gravity[0], gravity[1], gravity[2])
    get_b(v, b, f)
    matmul_edge(mul_ans, v, edges) #mul_ans = A*v0    
    add(r0, b, -1, mul_ans) # r0 = b-Ax0 = b-mul_ans
    r02mgvector(mg_r0, r0)
    Multigrid_pre()
    mgvector2r0(z0, mg_z0)
    matmul_edge(cr_mul_ans, z0, edges) #cr_mul_ans = A*z0
    ndarray_to_ndarray(p0, z0) # p0 = z0
    dot2scalar(r0, cr_mul_ans) # dot_ans = r0*A*z0
    init_r_2() # r_2_scalar = dot_ans
    
    for CG_ITERS in range(30):
        
        matmul_edge(mul_ans, p0, edges) #mul_ans = A*p0
        dot2scalar(mul_ans, mul_ans) #dot_ans = A*p0*A*p0
        update_alpha(alpha_scalar, r_2_scalar, dot_ans) #alpha=r_2_scalar/dot_ans
        add_scalar_ndarray(v, v, 1, alpha_scalar, p0) #vn = vn+alpha*p0
        add_scalar_ndarray(r0, r0, -1, alpha_scalar, mul_ans) #r0 = r0-alpha*A*p0
        r02mgvector(mg_r0, r0)
        Multigrid_pre()
        mgvector2r0(z0, mg_z0)
        matmul_edge(cr_mul_ans, z0, edges) #mul_ans = A*z0
        dot2scalar(r0, cr_mul_ans) #dot_ans = r0*A*z0
        update_beta_r_2(beta_scalar, dot_ans, r_2_scalar) #beta=dot_ans/r_2_sca, r_2_sca=dot_ans
        add_scalar_ndarray(p0, z0, 1, beta_scalar, p0)
        
    fill_ndarray(f, 0)
    add(x, x, dt, v)

def Jacobi_pcg(it):

    get_force(x, f, vertices, gravity[0], gravity[1], gravity[2])
    get_b(v, b, f)
    matmul_edge(mul_ans, v, edges) #mul_ans = A*v0    
    add(r0, b, -1, mul_ans) # r0 = b-Ax0 = b-mul_ans
    Jacobi_pre(z0, r0)
    ndarray_to_ndarray(p0, z0) # p0 = z0
    dot2scalar(r0, z0) # dot_ans = r0*z0
    init_r_2() # r_2_scalar = dot_ans
        
    for CG_ITERS in range(num_substeps):
        
        matmul_edge(mul_ans, p0, edges) #mul_ans = A*p0
        dot2scalar(p0, mul_ans) #dot_ans = p0*A*p0
        update_alpha(alpha_scalar, r_2_scalar, dot_ans) #alpha=r_2_scalar/dot_ans
        add_scalar_ndarray(v, v, 1, alpha_scalar, p0) #vn = vn+alpha*p0
        add_scalar_ndarray(r0, r0, -1, alpha_scalar, mul_ans) #r0 = r0-alpha*A*p0
        Jacobi_pre(z0, r0) #z0=M^-1*r0
        dot2scalar(r0, z0) #dot_ans = r0*z0
        update_beta_r_2(beta_scalar, dot_ans, r_2_scalar) #beta=dot_ans/r_2_sca, r_2_sca=dot_ans
        add_scalar_ndarray(p0, z0, 1, beta_scalar, p0)
        
    fill_ndarray(f, 0)
    add(x, x, dt, v)


def Jacobi_pcr(it):

    get_force(x, f, vertices, gravity[0], gravity[1], gravity[2])
    get_b(v, b, f)
    matmul_edge(mul_ans, v, edges) #mul_ans = A*v0    
    add(r0, b, -1, mul_ans) # r0 = b-Ax0 = b-mul_ans
    Jacobi_pre(z0, r0) #obtain z0
    matmul_edge(cr_mul_ans, z0, edges) #cr_mul_ans = A*z0
    ndarray_to_ndarray(p0, z0) # p0 = z0
    dot2scalar(r0, cr_mul_ans) # dot_ans = r0*A*z0
    init_r_2() # r_2_scalar = dot_ans
    
    for CG_ITERS in range(20):
        
        matmul_edge(mul_ans, p0, edges) #mul_ans = A*p0
        dot2scalar(mul_ans, mul_ans) #dot_ans = A*p0*A*p0
        update_alpha(alpha_scalar, r_2_scalar, dot_ans) #alpha=r_2_scalar/dot_ans
        add_scalar_ndarray(v, v, 1, alpha_scalar, p0) #vn = vn+alpha*p0
        add_scalar_ndarray(r0, r0, -1, alpha_scalar, mul_ans) #r0 = r0-alpha*A*p0

        itererror(z0) #res = r1.norm()
        print("residual:", CG_ITERS, res_scalar[None])
        
        
        matmul_edge(cr_mul_ans, z0, edges) #mul_ans = A*z0
        dot2scalar(r0, cr_mul_ans) #dot_ans = r0*A*z0
        update_beta_r_2(beta_scalar, dot_ans, r_2_scalar) #beta=dot_ans/r_2_sca, r_2_sca=dot_ans
        add_scalar_ndarray(p0, z0, 1, beta_scalar, p0)
        
    fill_ndarray(f, 0)
    add(x, x, dt, v)



def cg(it):

    get_force(x, f, vertices, gravity[0], gravity[1], gravity[2])
    get_b(v, b, f)
    matmul_edge(mul_ans, v, edges) #mul_ans = A*v0
    add(r0, b, -1, mul_ans) # r0 = b-Av0 = b-mul_ans
    ndarray_to_ndarray(p0, r0) # p0 = r0
    dot2scalar(r0, r0) # dot_ans = r0*r0
    init_r_2() # r_2_scalar = dot_ans
    
    for CG_ITERS in range(30):
        matmul_edge(mul_ans, p0, edges) #mul_ans = A*p0
        dot2scalar(p0, mul_ans) #dot_ans = p0*A*p0
        update_alpha(alpha_scalar, r_2_scalar, dot_ans) #alpha
        add_scalar_ndarray(v, v, 1, alpha_scalar, p0) #vn = vn+alpha*p0
        add_scalar_ndarray(r0, r0, -1, alpha_scalar, mul_ans) #r0 = r0-alpha*A*p0
        dot2scalar(r0, r0) #dot_ans = r0*r0
        update_beta_r_2(beta_scalar, dot_ans, r_2_scalar) #beta
        add_scalar_ndarray(p0, r0, 1, beta_scalar, p0)        
        
    fill_ndarray(f, 0)
    add(x, x, dt, v)


# 感觉这个residual算得就很有问题？？很奇怪？？到底是怎么算的？？为什么residual降不下去
def cr(it):
    get_force(x, f, vertices, gravity[0], gravity[1], gravity[2])
    get_b(v, b, f)
    matmul_edge(mul_ans, v, edges) #mul_ans = A*v0
    add(r0, b, -1, mul_ans) # r0 = b-Ax0 = b-mul_ans
    matmul_edge(cr_mul_ans, r0, edges) #cr_mul_ans = A*r0 
    ndarray_to_ndarray(p0, r0) # p0 = r0
    dot2scalar(r0, cr_mul_ans) # dot_ans = r0*A*r0
    init_r_2() # r_2_scalar = dot_ans = r0*A*r0

    for CG_ITERS in range(30):
        matmul_edge(mul_ans, p0, edges) #mul_ans = A*p0
        dot2scalar(mul_ans, mul_ans) #dot_ans = (A*p0)*(A*p0)
        update_alpha(alpha_scalar, r_2_scalar, dot_ans) #alpha= r_2_scalar[None] / (dot_ans[None] + epsilon)
        add_scalar_ndarray(v, v, 1, alpha_scalar, p0) #vn = vn+alpha*p0
        add_scalar_ndarray(r0, r0, -1, alpha_scalar, mul_ans) #r1 = r0-alpha*(A*p0)

        # itererror(r0) #res = r1.norm()
        # print("residual:", CG_ITERS, res_scalar[None])
        print(CG_ITERS, r0[10])


        matmul_edge(cr_mul_ans, r0, edges) #mul_ans = A*r0
        dot2scalar(r0, cr_mul_ans) #dot_ans = r1*A*r1
        update_beta_r_2(beta_scalar, dot_ans, r_2_scalar) #beta=(r1*A*r1)/(r0*A*r0)
        add_scalar_ndarray(p0, r0, 1, beta_scalar, p0)

    fill_ndarray(f, 0)
    add(x, x, dt, v)


def save_ply(frame_id):
    series_prefix = "pcr_armodilo.ply"
    writer = ti.tools.PLYWriter(num_vertices=n_verts, num_faces=n_faces, face_type="tri")
    np_pos = np.reshape(x.to_numpy(), (n_verts, 3))
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    np_indices = indices_np.reshape(-1)
    writer.add_faces(np_indices)
    writer.export_frame_ascii(frame_id, series_prefix)


@ti.kernel
def init(x: ti.types.ndarray(), v: ti.types.ndarray(), f: ti.types.ndarray(),
         ox: ti.types.ndarray(), vertices: ti.types.ndarray()):
    for u in x:
        x[u] = ox[u]
        v[u] = [0.0] * 3
        f[u] = [0.0] * 3
        m[u] = 0.0
    for c in vertices:
        F = Ds(vertices[c], x)
        B[c] = F.inverse()
        W[c] = ti.abs(F.determinant()) / 6
        for i in ti.static(range(4)):
            m[vertices[c][i]] += W[c] / 4 * density


@ti.kernel
def floor_bound(x: ti.types.ndarray(), v: ti.types.ndarray()):
    bounds = ti.Vector([1, 1, 1])
    for u in x:
        for i in ti.static(range(3)):
            if x[u][i] < -bounds[i]:
                x[u][i] = -bounds[i]
                if v[u][i] < 0:
                    v[u][i] = 0
            if x[u][i] > bounds[i]:
                x[u][i] = bounds[i]
                if v[u][i] > 0:
                    v[u][i] = 0


def substep():
    for i in range(num_substeps):
        #MG_pcr(i)
        #MG_pcg(i)
        # Jacobi_pcr(i)
        #Jacobi_pcg(i)
        #cg(i)
        cr(i)
    floor_bound(x, v)


@ti.kernel
def convert_to_field(x: ti.types.ndarray(), y: ti.template()):
    for I in ti.grouped(x):
        y[I] = x[I]



def run_ggui():
    res = (600, 600)
    window = ti.ui.Window("Implicit FEM", res, vsync=True)

    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 1.5, 2.95)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(55)

    x_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

    def handle_interaction(window):
        global gravity
        gravity = [0, -9.8, 0]
        if window.is_pressed('i'): gravity = [0, 0, -9.8]
        if window.is_pressed('k'): gravity = [0, 0, 9.8]
        if window.is_pressed('o'): gravity = [0, 9.8, 0]
        if window.is_pressed('u'): gravity = [0, -9.8, 0]
        if window.is_pressed('l'): gravity = [9.8, 0, 0]
        if window.is_pressed('j'): gravity = [-9.8, 0, 0]

    def render():
        convert_to_field(x, x_field)
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.1, ) * 3)
        scene.point_light(pos=(0.5, 10.0, 0.5), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(10.0, 10.0, 10.0), color=(0.5, 0.5, 0.5))
        scene.mesh(x_field, indices, color=(0.73, 0.33, 0.73))
        canvas.scene(scene)

    frame = 0
    while window.running:

        substep()
        if window.is_pressed('r'):
            init(x, v, f, ox, vertices)
        if window.is_pressed(ti.GUI.ESCAPE):
            break
        handle_interaction(window)
        render()
        window.show()

        frame += 1


if __name__ == '__main__':
    
    print(f'dt={dt} num_substeps={num_substeps}')    
    clear_field()
    init(x, v, f, ox, vertices)
    get_matrix(c2e, vertices)
    run_ggui()
