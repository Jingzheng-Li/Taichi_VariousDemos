
# TODO：等下找一些那些是可以用gridm提前拿出来的

import taichi as ti
import taichi.math as tm
import numpy as np
import taichi_math_util_funcs as ufuncs

@ti.data_oriented
class MATH_LINEAR_SOLVER:
    def __init__(self, vertices, num_particles, n_grid, inv_dx, dt):

        self.vertices = vertices
        self.num_faces = vertices.shape[0]
        self.num_particles = num_particles
        self.inv_dx = inv_dx
        self.dt = dt

        self.grid_rhs = ti.Vector.field(3, ti.f32, (n_grid, n_grid, n_grid))
        self.grid_r0 = ti.Vector.field(3, ti.f32, (n_grid, n_grid, n_grid))
        self.grid_z0 = ti.Vector.field(3, ti.f32, (n_grid, n_grid, n_grid))
        self.grid_p0 = ti.Vector.field(3, ti.f32, (n_grid, n_grid, n_grid))
        self.gridv_guess = ti.Vector.field(3, ti.f32, (n_grid, n_grid, n_grid))

        self.alpha_scalar = ti.field(ti.f32, ())
        self.beta_scalar = ti.field(ti.f32, ())
        self.dot_scalar = ti.field(ti.f32, ())
        self.r_2_scalar = ti.field(ti.f32, ())
        self.res_scalar = ti.field(ti.f32, ())
        
        self.grid_Hessmul = ti.Vector.field(3, ti.f32, (n_grid, n_grid, n_grid)) #conjugate gradient
        self.grid_crHessmul = ti.Vector.field(3, ti.f32, (n_grid, n_grid, n_grid)) #conjugate residual

        self.grid_mulbuffer = ti.Vector.field(3, ti.f32, self.num_particles) # 
        self.ptcl_mulbuffer = ti.Vector.field(3, ti.f32, self.num_particles)


    @ti.kernel
    def add(self, ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
        for I in ti.grouped(ans):
            ans[I] = a[I] + k * b[I]

    @ti.kernel
    def add_scalar_field(self, ans: ti.template(), a: ti.template(),
                        k: ti.f32, scalar: ti.template(),
                        b: ti.template()):
        for I in ti.grouped(ans):
            ans[I] = a[I] + k * scalar[None] * b[I]

    @ti.kernel
    def dot2scalar(self, a: ti.template(), b: ti.template()):
        self.dot_scalar[None] = 0.0
        for I in ti.grouped(a):
            self.dot_scalar[None] += a[I].dot(b[I])


    @ti.kernel
    def field_to_field(self, field: ti.template(), other: ti.template()):
        for I in ti.grouped(field):
            field[I] = other[I]

    @ti.kernel
    def init_r_2(self, ):
        self.r_2_scalar[None] = self.dot_scalar[None]


    @ti.kernel
    def update_alpha(self, alpha_scalar: ti.template(),  numerator: ti.template(), denominator: ti.template()):
        alpha_scalar[None] = numerator[None] / (denominator[None] + 1e-8)

    @ti.kernel
    def update_beta_r_2(self, beta_scalar: ti.template(), numerator: ti.template(), denominator: ti.template()):
        beta_scalar[None] = numerator[None] / (denominator[None] + 1e-8)
        denominator[None] = numerator[None]


    @ti.kernel
    def get_rhs(self, grid_rhs:ti.template(), gridv:ti.template()):
        for I in ti.grouped(grid_rhs):
            if gridv[I].any():
                grid_rhs[I] = gridv[I]

    # 这个用gridv.any是不是可以的呢？
    @ti.kernel
    def get_gridv_guess(self, gridv_guess: ti.template(), gridv: ti.template(), gridm: ti.template()):
        for I in ti.grouped(gridv):
            if gridv[I].any():
                gridv_guess[I] = gridv[I] / gridm[I]


    # TODO: 这个地方是需要大改的地方 我才发现 这个地方是不是有什么问题？？？怎么能都是gridv呢？？？命名是Av0Ar0啊？？
    @ti.kernel
    def get_Hessmul(self, grid_Hessmul:ti.template(), grid_ans:ti.template(), gridm:ti.template(),
                            Hesstotal:ti.template(), ptclx:ti.template(), ):
        
        # grid_Hessmul = (M+dt^2*Hesstotal)*grid_ans = A*v0/A*p0/A*r0
        self.grid_mulbuffer.fill(0)
        self.ptcl_mulbuffer.fill(0)
        grid_Hessmul.fill(0)

        for p in range(self.num_particles):
            base, fx, w = ufuncs.B_spline_weight(ptclx[p], self.inv_dx)
            new_ans = ti.Vector.zero(float, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                weight = w[i,0] * w[j,1] * w[k,2]
                g_ans = grid_ans[base + offset]
                new_ans += weight * g_ans
            self.ptcl_mulbuffer[p] = new_ans


        for f in range(self.num_faces):
            a, b, c = ufuncs.points_of_face(f, self.vertices)
            vertsid = (a,b,c)
            for i,j in ti.static(ti.ndrange(3,3)):
                self.grid_mulbuffer[vertsid[i]] += Hesstotal[f,i,j] @ self.ptcl_mulbuffer[vertsid[j]]

        for p in range(self.num_particles):
            base, fx, w = ufuncs.B_spline_weight(ptclx[p], self.inv_dx)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                weight = w[i,0] * w[j,1] * w[k,2]
                # H*grid_ans
                grid_Hessmul[base + offset] += weight * self.grid_mulbuffer[p]

        for I in ti.grouped(grid_ans):
            if grid_ans[I].any():
                grid_Hessmul[I] *= self.dt**2
                grid_Hessmul[I] += gridm[I] * grid_ans[I]

    @ti.kernel
    def iter_error(self, a: ti.template(),):
        self.res_scalar[None] = 0.0
        for I in ti.grouped(a):
            self.res_scalar[None] += a[I].norm_sqr()
        self.res_scalar[None] = ti.sqrt(self.res_scalar[None])


    def substep_conjuagte_residual(self, iter, ptclx, gridv, gridm, Hesstotal):

        self.get_rhs(self.grid_rhs, gridv) # grid_rhs = gridv = b
        self.get_gridv_guess(self.gridv_guess, gridv, gridm) # gridv_gauss = gridv/gridm

        self.get_Hessmul(self.grid_Hessmul, self.gridv_guess, gridm, Hesstotal, ptclx) # grid_Hessmul = A*v0
        self.add(self.grid_r0, self.grid_rhs, -1, self.grid_Hessmul) # r0 = b-Av0 = b-grid_Hessmul
        self.get_Hessmul(self.grid_Hessmul, self.grid_r0, gridm, Hesstotal, ptclx) # grid_crHessmul = A*r0
        self.field_to_field(self.grid_p0, self.grid_r0) # p0 = r0
        self.dot2scalar(self.grid_r0, self.grid_crHessmul) # dot_scalar = r0*A*r0
        self.init_r_2() # r_2_scalar = dot_scalar = r0*A*r0

        for i in range(iter):
            self.get_Hessmul(self.grid_Hessmul, self.grid_p0, gridm, Hesstotal, ptclx) # grid_Hessmul = A*p0
            self.dot2scalar(self.grid_Hessmul, self.grid_Hessmul) # dot_scalar = (A*p0)*(A*p0)
            self.update_alpha(self.alpha_scalar, self.r_2_scalar, self.dot_scalar) # alpha= r_2_scalar[None] / (dot_scalar[None] + epsilon)
            self.add_scalar_field(self.gridv_guess, self.gridv_guess, 1, self.alpha_scalar, self.grid_p0) #vn = vn+alpha*p0
            self.add_scalar_field(self.grid_r0, self.grid_r0, -1, self.alpha_scalar, self.grid_Hessmul) #r1 = r0-alpha*(A*p0)

            # TODO：为什么我的cr下降这么慢？？还是很奇怪
            self.iter_error(self.grid_r0) #res = r1.norm()
            if self.res_scalar[None] < 1e-3:
                # print("residual:", i, res_scalar[None])
                break

            self.get_Hessmul(self.grid_Hessmul, self.grid_r0, gridm, Hesstotal, ptclx) #mul_ans = A*r1
            self.dot2scalar(self.grid_r0, self.grid_crHessmul) #dot_scalar = r1*A*r1
            self.update_beta_r_2(self.beta_scalar, self.dot_scalar, self.r_2_scalar) #beta=(r1*A*r1)/(r0*A*r0)
            self.add_scalar_field(self.grid_p0, self.grid_r0, 1, self.beta_scalar, self.grid_p0) #p1=r1+beta*p0

        # print("residual:", res_scalar[None])
        return self.gridv_guess, self.res_scalar[None]


    # 为什么还是不对 真的有点着急了 把别人的规整一下也比自己写来的好。。要不然换路线把 我真的有点累了
    def substep_pre_conjuate_residual(self, iter, ptclx, gridv, gridm, Hesstotal):
        self.get_rhs(self.grid_rhs, gridv) # grid_rhs = gridv = b
        self.get_gridv_guess(self.gridv_guess, gridv, gridm) # gridv_gauss = gridv/gridm








