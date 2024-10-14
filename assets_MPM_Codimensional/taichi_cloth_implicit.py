

import taichi as ti
import taichi.math as tm
import numpy as np
import taichi_math_util_funcs as ufuncs
import os


import taichi_math_linear_solver
import taichi_make_universal_gravity
from ClothForce import taichi_make_cloth_membrane
from ClothForce import taichi_make_cloth_bend
from ClothForce import taichi_make_cloth_collision



ti.init(arch = ti.gpu, device_memory_fraction=0.8, debug=False)
# ti.init(arch = ti.cpu, device_memory_fraction=0.7, debug=True)
# ti.init(arch = ti.cpu, device_memory_fraction=0.7, debug=True, cpu_max_num_threads=1)

# simulate paramters
n_grid = 64 # 128 for large one, 64 for small one
dx = 1.0 / n_grid
inv_dx = 1.0 / dx
n_substeps = 60
dt = 1.0/24/n_substeps

# compression tension shear tension bending

# Material Parameters
coef_density = 1.0 # particle density
coef_normalstiff = 500.0 # stiffness k 法线方向compression 碰撞回弹度
coef_shearstiff = 0.0 # shearstiffness gamma 法线方向shearing 形变程度
coef_damping = 0.8 # damping cofficient
coef_fric = 0.2 # friction cofficient
coef_Young = 2e4 # E youngs modulus
coef_Poisson = 0.3 # eta Poisson ratio
coef_Visc = 5e3 # viscous modulus
coef_thickness = 0.04 # thickness
coef_bendstiff = 0.1 # force to resist bend
coef_viscstiff = 0.1 # force to average motion 运动黏度 viscous stiff

cloth_membrane_param = [coef_Young, coef_Poisson, coef_Visc, coef_thickness]
cloth_bend_param = [coef_bendstiff, coef_viscstiff]
cloth_collision_param = [coef_normalstiff, coef_shearstiff, coef_fric]


# Lame parameter & other parameter
lame_mu = coef_Young/(2.0*(1.0+coef_Poisson)) # lame parameter
lame_la = coef_Young*coef_Poisson/((1.0+coef_Poisson)*(1.0-2.0*coef_Poisson)) # lame parameter



def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # 仅提取前三个坐标（x, y, z）
                vertex = [float(p) for p in parts[1:4]]
                vertices.append(vertex)
            elif line.startswith('f '):
                parts = line.strip().split()
                face = []
                for part in parts[1:4]:
                    # 处理如 'f 1/1/1 2/2/2 3/3/3' 的情况，只取顶点索引
                    idx = part.split('/')[0]
                    face.append(int(idx) - 1)  # OBJ索引从1开始，转为0基
                faces.append(face)
                
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)




# 加载OBJ文件并处理
obj_filename = "assets_geometry/64x64cloth.obj"  # 替换为你的OBJ文件路径
ptclrestxnp, verticesnp = load_obj(obj_filename)


ptclrestx = ti.Vector.field(3, ti.f32, ptclrestxnp.shape[0])
ptclrestx.from_numpy(ptclrestxnp)
vertices = ti.Vector.field(3, ti.i32, verticesnp.shape[0])
vertices.from_numpy(verticesnp)
num_particles = ptclrestx.shape[0] # 16384*2
num_faces = vertices.shape[0] # 32258*2

# particles on cloth
ptclx = ti.Vector.field(3, ti.f32, num_particles) # position
ptclv = ti.Vector.field(3, ti.f32, num_particles) # velocity
ptclC = ti.Matrix.field(3, 3, ti.f32, num_particles) # affine velocity field
ptclrestvol = ti.field(ti.f32, num_particles)
ptclmass = ti.field(ti.f32, num_particles)

# faces of cloth
facex = ti.Vector.field(3, ti.f32, num_faces) # position
facev = ti.Vector.field(3, ti.f32, num_faces) # velocity
faceC = ti.Matrix.field(3, 3, ti.f32, num_faces) # affine velocity field
facerestvol = ti.field(ti.f32, num_faces) # volume of each triangle
facemass = ti.field(ti.f32, num_faces) # mass of each triangle

# implicit of cloth
forcetotal = ti.Vector.field(3, ti.f32, num_particles)
Hesstotal = ti.Matrix.field(3,3, ti.f32, (num_faces, 3, 3))


# grid
gridv = ti.Vector.field(3, ti.f32, (n_grid, n_grid, n_grid))
gridm = ti.field(ti.f32, (n_grid, n_grid, n_grid))

# draw scenes
clothindices = ti.field(ti.i32, num_faces*3)
clothcolors = ti.Vector.field(3, ti.f32, num_faces)

# collision paramters
ball_center = ti.Vector([0.51,0.6,0.51])
ball_radius = 0.08
bound = 3 # boundary


universal_graivity = taichi_make_universal_gravity.MAKE_UNIVERSAL_GRAVITY(
    tm.vec3(0,-9.8,0), ptclmass)


cloth_membrane = taichi_make_cloth_membrane.MAKE_CLOTH_MEMBRANE(
    vertices, ptclrestx, ptclrestvol, facerestvol, cloth_membrane_param)

cloth_bend = taichi_make_cloth_bend.MAKE_CLOTH_BEND(
    vertices, ptclrestx, ptclrestvol, facerestvol, cloth_bend_param)

cloth_collision = taichi_make_cloth_collision.MAKE_CLOTH_COLLISION(
    vertices, ptclrestx, ptclrestvol, facerestvol, cloth_collision_param)


linsolver = taichi_math_linear_solver.MATH_LINEAR_SOLVER(
    vertices, num_particles, n_grid, inv_dx, dt)




@ti.kernel
def initialize_mesh_points():
    for p in range(num_particles):
        ptclx[p] = ptclrestx[p]
        ptclv[p] = tm.vec3(0.0)
        ptclC[p] = tm.mat3(0.0)

    for f in range(num_faces):
        a, b, c = ufuncs.points_of_face(f, vertices)
        clothindices[3*f], clothindices[3*f+1], clothindices[3*f+2] = a, b, c
        DE0, DE1 = ptclrestx[b]-ptclrestx[a], ptclrestx[c]-ptclrestx[a] # fem mesh part
        DE1_cross_DE0 = tm.cross(DE1, DE0)

        facex[f] = (ptclrestx[a] + ptclrestx[b] + ptclrestx[c])/3.0
        facev[f] = tm.vec3(0.0)
        faceC[f] = tm.mat3(0.0)

        # TODO: 修改vol成volumuefraction的方式
        facerestarea = 0.5 * DE1_cross_DE0.norm()
        unitvol = coef_thickness * facerestarea
        ptclrestvol[a] += unitvol # shared points with larger volume&mass
        ptclrestvol[b] += unitvol
        ptclrestvol[c] += unitvol
        facerestvol[f] = unitvol
        # facerestvol[f] = 4.0 * unitvol
        ptclmass[a] += coef_density * unitvol
        ptclmass[b] += coef_density * unitvol
        ptclmass[c] += coef_density * unitvol
        facemass[f] = coef_density * unitvol

    # collision force between codimension
    cloth_collision.initialize_cloth_collision()

    # thin shell membrane force
    cloth_membrane.initialize_cloth_membrane()

    # thin shell bending force
    cloth_bend.initialize_cloth_bend()
    cloth_bend.get_edgeshared_phi(ptclrestx, cloth_bend.edgeshared_restphi)




@ti.kernel
def substep_update_startstate():
    gridv.fill(0)
    gridm.fill(0)
    forcetotal.fill(0)
    Hesstotal.fill(0)


@ti.kernel
def substep_get_gravity_force():
    for p in ptclx:
        universal_graivity.get_gravity_force(p, forcetotal)


@ti.kernel
def substep_get_membrane_force():
    for f in facex:
        a, b, c = ufuncs.points_of_face(f, vertices)
        xa, xb, xc = ptclx[a], ptclx[b], ptclx[c]
        cloth_membrane.get_cloth_membrane_force(f, (a,b,c), (xa,xb,xc), forcetotal)
        cloth_membrane.get_cloth_membrane_Hessian(f, (a,b,c), (xa,xb,xc), Hesstotal)


@ti.kernel
def substep_get_bending_force():
    for e in cloth_bend.edgeshared:
        es = cloth_bend.edgeshared[e]
        idxvec = cloth_bend.edgeshared_ptclid[es]
        idx0, idx1, idx2, idx3 = idxvec[0], idxvec[1], idxvec[2], idxvec[3] # 四个顶点的索引
        x0, x1, x2, x3 = ptclx[idx0], ptclx[idx1], ptclx[idx2], ptclx[idx3] # 四个顶点的坐标
        cloth_bend.get_cloth_bend_force(es, (idx0, idx1, idx2, idx3), (x0, x1, x2, x3), forcetotal)


@ti.kernel
def substep_p2g_particles():
    for p in ptclx:
        base, fx, w = ufuncs.B_spline_weight(ptclx[p], inv_dx)
        mass_p = ptclmass[p]
        affine = ptclC[p] * mass_p

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i,0] * w[j,1] * w[k,2]
            dpos = (offset.cast(float) - fx) * dx

            # affine momentum 
            # velocity momentum
            # thin shell force momentum
            gridv[base + offset] += weight * (affine @ dpos) \
                                    + weight * (mass_p * ptclv[p]) \
                                    + weight * (forcetotal[p] * dt)

            # compute type2 mass
            gridm[base + offset] += weight * mass_p


@ti.kernel
def substep_p2g_faces():
    for f in facex:
        facevolume = facerestvol[f]
        base, fx, w = ufuncs.B_spline_weight(facex[f], inv_dx)
        dw_dx_d = ti.Matrix.rows([fx-1.5, 2*(1.0-fx), fx-0.5]) * inv_dx
        dPsi_dF2, dE2 = cloth_collision.get_cloth_collision_force(f)
        facemass_p = facemass[f]
        affine = facemass_p * faceC[f]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i,0] * w[j,1] * w[k,2] # 对于facex的权重
            dw_dx = ti.Vector([dw_dx_d[i,0] * w[j,1] * w[k,2], # facex权重求导
                               w[i,0] * dw_dx_d[j,1] * w[k,2],
                               w[i,0] * w[j,1] * dw_dx_d[k,2]])

            gridforce_normal = -facevolume * dPsi_dF2 * dw_dx.dot(dE2)
            dpos = (offset.cast(float) - fx) * dx
            
            # affine momentum + velocity momentum + collision force momentum
            gridv[base + offset] += weight * (affine @ dpos) \
                                    + weight * (facemass_p * facev[f]) \
                                    + (gridforce_normal * dt)

            # compute type3 facemass
            gridm[base + offset] += weight * facemass_p


@ti.kernel
def substep_update_collision(gridv_guess:ti.template()):
    for I in ti.grouped(gridm):
        if gridv_guess[I].any():
            gridvel = gridv_guess[I]

            # ball collision
            dist = I.cast(float)*dx - ball_center
            if  dist.norm() < 1.05 * ball_radius:
                dist = dist.normalized()
                gridvel -= dist * ti.min(0, gridvel.dot(dist))
                # gridv[I] *= 0.95  #friction

            cond = (I < bound) & (gridvel < 0) | (I > n_grid - bound) & (gridvel > 0)
            gridvel = ti.select(cond, 0, gridvel)
            gridv[I] = gridvel


def substep_update_grid_v():
    gridv_guess, res_scalar = linsolver.substep_conjuagte_residual(
        100, ptclx, gridv, gridm, Hesstotal)

    substep_update_collision(gridv_guess)




@ti.kernel
def substep_g2p_particles():
    for p in ptclx:
        base, fx, w = ufuncs.B_spline_weight(ptclx[p], inv_dx)
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = offset.cast(float) - fx
            weight = w[i,0] * w[j,1] * w[k,2]
            g_v = gridv[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        ptclv[p] = new_v
        ptclx[p] += dt * ptclv[p]
        # C_sym, C_skew = ufuncs.split_sym_skew(new_C)
        # ptclC[p] = C_skew + (1.0-coef_damping)*C_sym
        ptclC[p] = new_C


@ti.kernel
def substep_g2p_faces():
    for f in facex:
        a, b, c = ufuncs.points_of_face(f, vertices)
        base, fx, w = ufuncs.B_spline_weight(facex[f], inv_dx)
        new_C = tm.mat3(0.0)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = offset.cast(float) - fx
            weight = w[i,0] * w[j,1] * w[k,2]
            g_v = gridv[base + offset]
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        facev[f] = (ptclv[a] + ptclv[b] + ptclv[c]) / 3.0
        facex[f] = (ptclx[a] + ptclx[b] + ptclx[c]) / 3.0
        # C_sym, C_skew = ufuncs.split_sym_skew(new_C)
        # faceC[f] = C_skew + (1.0-coef_damping)*C_sym
        faceC[f] = new_C

@ti.kernel
def substep_return_mapping():
    for f in facex:
        cloth_collision.get_cloth_return_mapping(f, ptclx, dt, faceC)


def save_mesh_as_obj(frame_idx):
    output_dir = "output_mesh"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.obj")
    with open(filename, 'w') as f:
        # Write vertices
        for i in range(num_particles):
            f.write(f"v {ptclx[i][0]} {ptclx[i][1]} {ptclx[i][2]}\n")
        # Write faces
        for i in range(num_faces):
            a, b, c = vertices[i]
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")
    print(f"Saved frame {frame_idx} to {filename}")

def run_ggui():
    res = (800, 800)
    window = ti.ui.Window("taichi 3D cloth", res, vsync=False)

    canvas = window.get_canvas()
    canvas.set_background_color((1.0, 1.0, 1.0))
    scene = ti.ui.Scene()

    camera = ti.ui.Camera()
    camera.position(0.5, 1.2, 1.95)
    camera.lookat(0.5, 0.3, 0.5)
    camera.fov(55)

    square_bound = ti.Vector.field(3,ti.f32,8)
    square_bound[0],square_bound[1],square_bound[2] = [0,0,0],[1,0,0],[0,1,0]
    square_bound[3],square_bound[4],square_bound[5] = [0,0,1],[1,1,0],[0,1,1]
    square_bound[6],square_bound[7] = [1,0,1],[1,1,1]

    collision_ball = ti.Vector.field(3,ti.f32,1)
    collision_ball[0] = ball_center

    # initialize simulation here
    cloth_bend.get_cloth_bend_geometry()

    initialize_mesh_points()

    frame_idx = 0
    while window.running:

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.1, ) * 3)
        scene.point_light(pos=(0.5, 10.0, 0.5), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(10.0, 10.0, 10.0), color=(0.5, 0.5, 0.5))
        scene.particles(square_bound, radius=0.02, color=(0.0, 1.0, 1.0))
        scene.particles(collision_ball, radius=0.9*ball_radius, color=(0.0, 1.0, 1.0))
        scene.mesh(ptclx, indices=clothindices, two_sided=True, show_wireframe=False, color=(0.42, 0.72, 0.52))
        # scene.particles(x, radius=0.005, color=(0.0, 1.0, 0.0))
        # scene.particles(x3, radius=0.005, color=(0.0, 0.0, 1.0))
        # scene.particles(my_render_sdf, radius=0.002, per_vertex_color=my_render_sdfcolor)


        for _ in range(n_substeps):
            substep_update_startstate()

            substep_get_gravity_force()
            # substep_get_bending_force()
            substep_get_membrane_force()
            substep_p2g_particles()
            substep_p2g_faces()

            substep_update_grid_v()

            substep_g2p_particles()
            substep_g2p_faces()
            substep_return_mapping()
        
        canvas.scene(scene)
        window.show()
        
        # Save current frame as OBJ
        save_mesh_as_obj(frame_idx)
        frame_idx += 1


if __name__ == '__main__':
    
    run_ggui()