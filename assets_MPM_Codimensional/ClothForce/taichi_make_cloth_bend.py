# TODO: psi_coeff先定为restarea 后面再从volumefraction做修改
# TODO: viscosity 有点问题？不知道什么意思 我先把它扯下来

import taichi as ti
import taichi.math as tm
import numpy as np
import taichi_math_util_funcs as ufuncs

"""   
edgeuniq 所有边的集合(单独)
edgeuniq_faceid 一个edge对应的两个face的index(没有face的就是-1)
edgeuniq_vertid 一个edge对应的两个vertex的index(没有index的就是-1)
edgeshared 被两个face share的edge编号(不含有-1 两侧都是有face的)
edgeshared_ptclid: 一个tri pair对应四个点的index
edgeshared_faceid: 一个tri pair对应两个面的index
"""
@ti.data_oriented
class MAKE_CLOTH_BEND:
    def __init__(self, vertices, ptclrestx, ptclrestvol, facerestvol, cloth_bend_param):
        self.vertices = vertices
        self.ptclrestx = ptclrestx
        self.ptclrestvol = ptclrestvol
        self.facerestvol = facerestvol
        self.num_faces = vertices.shape[0]
        
        self.coef_bendstiff = cloth_bend_param[0]
        self.coef_viscstiff = cloth_bend_param[1]

        self.psi_coeff = 1.0 
                
    def get_edge_flaps(self, vertices):

        num_faces = vertices.shape[0]
        num_edgedir = 3 * num_faces
        edgedir = ti.Vector.field(2, ti.i32, num_edgedir) # 有向边
        edgedir_sort = ti.Vector.field(2, ti.i32, num_edgedir) # 排序后的有向边
        self.oriented_facets(vertices, edgedir, edgedir_sort) # 获得排列后的有向边 计算方式是每个face三条边相加

        # 获得全排列后的有向边 变成全排列的索引 去重后unique edge的索引
        edgedir_ssortnp, edgedir_ssortnp_ind, edgeuniqnp_ind = self.unique_rows(edgedir_sort)
        num_edges = edgeuniqnp_ind.shape[0]
        edgedir_ssort = ti.Vector.field(2, ti.i32, num_edgedir) # 全排序有向边
        edgedir_ssort_ind = ti.field(ti.i32, num_edgedir)
        edgeuniq_ind = ti.field(ti.i32, num_edges)
        edgedir_ssort.from_numpy(edgedir_ssortnp)
        edgedir_ssort_ind.from_numpy(edgedir_ssortnp_ind)
        edgeuniq_ind.from_numpy(edgeuniqnp_ind)

        edgeuniq_reind = ti.field(ti.i32, num_edges)
        edgeuniq = ti.Vector.field(2, ti.i32, num_edges)
        medgeap = ti.field(ti.i32, num_edgedir)
        # 获得所有单独边的集合  和有向边到unique edge的映射
        self.unique_simplices(edgedir, edgedir_ssort, edgedir_ssort_ind,
                            edgeuniq_ind, edgeuniq_reind,
                            edgeuniq, medgeap,)

        edgeuniq_faceid = ti.Vector.field(2, ti.i32, num_edges)
        edgeuniq_vertid = ti.Vector.field(2, ti.i32, num_edges)
        self.unique_flaps(vertices, edgeuniq, medgeap, edgeuniq_faceid, edgeuniq_vertid)

        return edgeuniq, edgeuniq_faceid, edgeuniq_vertid

    @ti.kernel
    def oriented_facets(self, vertices:ti.template(),
                        edgedir:ti.template(), 
                        edgedir_sort:ti.template(),):
        num_faces = vertices.shape[0]
        for f in vertices:
            edgedir[f][0] = vertices[f][1]
            edgedir[f][1] = vertices[f][2]
            edgedir[f+num_faces][0] = vertices[f][2]
            edgedir[f+num_faces][1] = vertices[f][0]
            edgedir[f+2*num_faces][0] = vertices[f][0]
            edgedir[f+2*num_faces][1] = vertices[f][1]
        for e in edgedir:
            edgeind_min = tm.min(edgedir[e][0], edgedir[e][1])
            edgeind_max = tm.max(edgedir[e][0], edgedir[e][1])
            edgedir_sort[e] = ti.Vector([edgeind_min, edgeind_max])

    def unique_rows(self, edgedir_sort):
        num_edgedir = edgedir_sort.shape[0]
        edgedir_sortnp = edgedir_sort.to_numpy()
        edgedir_ssortnp_ind = np.lexsort((edgedir_sortnp[:,1], edgedir_sortnp[:,0]))
        edgedir_ssortnp = np.zeros([num_edgedir, 2], dtype=int)
        for e in range(num_edgedir):
            edgedir_ssortnp[e] = edgedir_sortnp[edgedir_ssortnp_ind[e]]
        useless_uniq, edgeuniqnp_ind = np.unique(edgedir_ssortnp, axis=0, return_index=True)
        return edgedir_ssortnp, edgedir_ssortnp_ind, edgeuniqnp_ind

    @ti.kernel
    def unique_simplices(self, edgedir:ti.template(),
                        edgedir_ssort:ti.template(),
                        edgedir_ssort_ind:ti.template(),
                        edgeuniq_ind:ti.template(),
                        edgeuniq_reind:ti.template(),
                        edgeunique:ti.template(),
                        medgeap:ti.template(),):

        num_edgedir = edgedir.shape[0] # 所有的有向边
        num_edges = edgeunique.shape[0] # 所有的uniqueedge（有向边去重）

        j = 0
        ti.loop_config(serialize=True)
        for i in range(num_edgedir):
            if any(edgedir_ssort[edgeuniq_ind[j]] != edgedir_ssort[i]):
                j += 1
            medgeap[edgedir_ssort_ind[i]] = j

        for i in range(num_edges):
            edgeuniq_reind[i] = edgedir_ssort_ind[edgeuniq_ind[i]]
        for i in range(num_edges):
            edgeunique[i] = edgedir[edgeuniq_reind[i]]

    @ti.kernel
    def unique_flaps(self, vertices:ti.template(),
                    edgeunique:ti.template(),
                    medgeap:ti.template(),
                    edgeuniq_faceid:ti.template(),
                    edgeuniq_vertid:ti.template(),):
        edgeuniq_faceid.fill(-1)
        edgeuniq_vertid.fill(-1)
        for f in range(self.num_faces):
            for v in range(3):
                e = medgeap[v * self.num_faces + f]
                if vertices[f][(v+1)%3] == edgeunique[e][0] and vertices[f][(v+2)%3] == edgeunique[e][1]:
                    edgeuniq_faceid[e][0] = f
                    edgeuniq_vertid[e][0] = v
                else:
                    edgeuniq_faceid[e][1] = f
                    edgeuniq_vertid[e][1] = v

    @ti.kernel
    def get_num_edgeshared(self, edgeuniq_faceid:ti.template()) -> ti.i32:
        ind = 0
        num_edges = edgeuniq_faceid.shape[0]
        for e in range(num_edges): # 找到含有shared有两个面的edge
            if all(edgeuniq_faceid[e] != ti.Vector([-1,-1])):
                ti.atomic_add(ind, 1)
        return ind

    @ti.kernel
    def get_edgeshared(self, ):
        ind = 0
        num_edges = self.edgeuniq_faceid.shape[0]
        for e in range(num_edges):
            if all(self.edgeuniq_faceid[e] != ti.Vector([-1,-1])):
                old_ind = ti.atomic_add(ind, 1)
                self.edgeshared[old_ind] = e
                
    # 这一块就是需要修改的地方了
    @ti.kernel
    def get_edgeshared_faceid_ptclid(self, ):

        for e in self.edgeshared:
            es = self.edgeshared[e]
            lface, rface = self.edgeuniq_faceid[es][0], self.edgeuniq_faceid[es][1]
            lvert, rvert = self.edgeuniq_vertid[es][0], self.edgeuniq_vertid[es][1]
            idx0 = self.vertices[lface][lvert]
            idx1 = self.edgeuniq[es][0]
            idx2 = self.edgeuniq[es][1]
            idx3 = self.vertices[rface][rvert]
            self.edgeshared_faceid[es] = tm.ivec2(lface, rface)
            self.edgeshared_ptclid[es] = tm.ivec4(idx0, idx1, idx2, idx3)


    def get_cloth_bend_geometry(self, ):
        """
        获取cloth bend的几何信息
        """
        self.edgeuniq, self.edgeuniq_faceid, self.edgeuniq_vertid = self.get_edge_flaps(self.vertices)
        num_edgeuniq = self.edgeuniq.shape[0]
        num_edgeshared = self.get_num_edgeshared(self.edgeuniq_faceid)
        self.edgeshared = ti.field(ti.i32, num_edgeshared)
        self.get_edgeshared()

        self.edgeshared_faceid = ti.Vector.field(2, ti.i32, num_edgeuniq) # 每个tripair对应两个face编号
        self.edgeshared_ptclid = ti.Vector.field(4, ti.i32, num_edgeuniq) # 每个triangle pair对应的4个点的index
        self.get_edgeshared_faceid_ptclid()

        self.edgeshared_restphi = ti.field(ti.f32, num_edgeuniq) # 每条边的rest能量
        self.edgeshared_startphi = ti.field(ti.f32, num_edgeuniq) # 每条边的start能量
        self.bending_ka = ti.field(ti.f32, num_edgeuniq)
        self.bending_kb = ti.field(ti.f32, num_edgeuniq)


    @ti.func
    def get_edgeshared_phi(self, ptclx:ti.template(), edgeshared_phi:ti.template(),):
        """
        获取cloth此时bend的程度带来的能量
        """
        for e in self.edgeshared:
            es = self.edgeshared[e]
            (idx0, idx1, idx2, idx3) = self.edgeshared_ptclid[es]

            x0 = ptclx[idx0]
            x1 = ptclx[idx1]
            x2 = ptclx[idx2]
            x3 = ptclx[idx3]

            medge, ledge, redge = x2 - x1, x0 - x2, x3 - x2
            lfnormal = tm.cross(medge, ledge)
            rfnormal = -tm.cross(medge, redge)
            lfnormal = lfnormal.normalized()
            rfnormal = rfnormal.normalized()

            # 计算一条normal和edge之间的夹角
            tmp_phi = tm.atan2(tm.dot(tm.cross(lfnormal, rfnormal), medge.normalized()), 
                               tm.dot(lfnormal,rfnormal))
            edgeshared_phi[es] = tmp_phi

    # TODO: 我为什么没有办法在这里更新psicoeff？
    @ti.func
    def initialize_cloth_bend(self, ):
        for e in self.edgeshared:
            es = self.edgeshared[e]
            idxvec = self.edgeshared_ptclid[es]
            idx0, idx1, idx2, idx3 = idxvec[0], idxvec[1], idxvec[2], idxvec[3]
            lface, rface = self.edgeshared_faceid[es][0], self.edgeshared_faceid[es][1]
            restareas = self.facerestvol[lface] + self.facerestvol[rface]
            restedgelen_square = ((self.ptclrestx[idx2] - self.ptclrestx[idx1]).norm())**2

            psi_coeff = self.psi_coeff * restareas
            self.bending_ka[es] = psi_coeff * self.coef_bendstiff * 3.0 * restedgelen_square / restareas
            # self.bending_kb[es] = psi_coeff * self.coef_viscstiff * 3.0 * restedgelen_square / restareas
            self.bending_kb[es] = 0.0


    # 现在问题就是我们怎么把数据传下来而且不出错？？
    # 想到两个 self和return先用self测试一下 return一定可以的
    @ti.func
    def get_cloth_bend_force(self, es, vertsid, vertspos, forcetotal):
        
        (idx0, idx1, idx2, idx3) = vertsid
        (x0, x1, x2, x3) = vertspos

        medge = x2 - x1
        medge_len = medge.norm()
        medge_normalized = medge / medge_len
        ledge0, ledge1 = x0 - x2, x0 - x1
        redge0, redge1 = x3 - x2, x3 - x1

        lfnormal = tm.cross(medge, ledge0)
        rfnormal = -tm.cross(medge, redge0)
        lfnormal_len = lfnormal.norm()
        rfnormal_len = rfnormal.norm()
        lfnormal /= lfnormal_len
        rfnormal /= rfnormal_len

        restphi = self.edgeshared_restphi[es]
        startphi = self.edgeshared_startphi[es]
        # 随着bending角度增大 currphi会变小
        currphi = tm.atan2(tm.dot(tm.cross(lfnormal,rfnormal), medge.normalized()),
                            tm.dot(lfnormal,rfnormal))
        
        ka, kb = self.bending_ka[es], self.bending_kb[es]
        dPsi_dTheta = 2.0 * (ka * (currphi - restphi) + kb * (currphi - startphi))
        
        # gridforce1/4是edge对应顶点的force（沿着facenormal方向） gridforce2/3是edge两个端点的force（）
        gridforce_1 = -dPsi_dTheta * (-medge_len / lfnormal_len * lfnormal)
        gridforce_2 = -dPsi_dTheta * (-tm.dot(medge_normalized, ledge0) / lfnormal_len * lfnormal +
                                    -tm.dot(medge_normalized, redge0) / rfnormal_len * rfnormal)
        gridforce_3 = -dPsi_dTheta * (tm.dot(medge_normalized, ledge1) / lfnormal_len * lfnormal +
                                    tm.dot(medge_normalized, redge1) / rfnormal_len * rfnormal)
        gridforce_4 = -dPsi_dTheta * (-medge_len / rfnormal_len * rfnormal)

        forcetotal[idx0] += gridforce_1
        forcetotal[idx1] += gridforce_2
        forcetotal[idx2] += gridforce_3
        forcetotal[idx3] += gridforce_4



"""

SOME BACKUP CODE FOR CHANGE FOR LOOP TO NUM_FACE

edgeshared_ownedfaceid, edgeshared_used = cloth_bend.get_cloth_bend_geometry()

print(edgeshared_ownedfaceid)

for f in facex:
    base_0 = ti.Matrix.zero(ti.i32,3,3) # 每一列存face对应三条边带来的base
    base_1 = ti.Matrix.zero(ti.i32,3,3)
    base_2 = ti.Matrix.zero(ti.i32,3,3)
    base_3 = ti.Matrix.zero(ti.i32,3,3)
    bend_force_0 = ti.Matrix.zero(ti.f32,3,3) # 每一列存face对应三条边带来的force
    bend_force_1 = ti.Matrix.zero(ti.f32,3,3)
    bend_force_2 = ti.Matrix.zero(ti.f32,3,3)
    bend_force_3 = ti.Matrix.zero(ti.f32,3,3)
    w_0 = ti.Matrix.zero(ti.f32,3,9) # 0-2列对应edge1 3-5列对应edge2 6-8列对应edge3
    w_1 = ti.Matrix.zero(ti.f32,3,9)
    w_2 = ti.Matrix.zero(ti.f32,3,9)
    w_3 = ti.Matrix.zero(ti.f32,3,9)

    for l in ti.static(range(3)):
        ef = edgeshared_ownedfaceid[f][l]
        if ef != -1:
            if edgeshared_used[ef] == 0: # 这个边是edgeshared 且从来没有被计算过
                idxvec = cloth_bend.edgeshared_ptclid[ef]
                idx0, idx1, idx2, idx3 = idxvec[0], idxvec[1], idxvec[2], idxvec[3]
                x0, x1, x2, x3 = ptclx[idx0], ptclx[idx1], ptclx[idx2], ptclx[idx3]
                base_0[:,l], fx_0, w_0_tmp = ufuncs.B_spline_weight(x0, inv_dx)
                base_1[:,l], fx_1, w_1_tmp = ufuncs.B_spline_weight(x1, inv_dx)
                base_2[:,l], fx_2, w_2_tmp = ufuncs.B_spline_weight(x2, inv_dx)
                base_3[:,l], fx_3, w_3_tmp = ufuncs.B_spline_weight(x3, inv_dx)
                w_0[:,3*l], w_0[:,3*l+1], w_0[:,3*l+2] = w_0_tmp[:,0], w_0_tmp[:,1], w_0_tmp[:,2]
                w_1[:,3*l], w_1[:,3*l+1], w_1[:,3*l+2] = w_1_tmp[:,0], w_1_tmp[:,1], w_1_tmp[:,2]
                w_2[:,3*l], w_2[:,3*l+1], w_2[:,3*l+2] = w_2_tmp[:,0], w_2_tmp[:,1], w_2_tmp[:,2]
                w_3[:,3*l], w_3[:,3*l+1], w_3[:,3*l+2] = w_3_tmp[:,0], w_3_tmp[:,1], w_3_tmp[:,2]

                (bend_force_0[:,l], bend_force_1[:,l], bend_force_2[:,l], bend_force_3[:,l]) = \
                cloth_bend.get_cloth_bend_force(ef, (idx0, idx1, idx2, idx3), (x0, x1, x2, x3))

                ti.atomic_add(edgeshared_used[ef], 1)

    for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
        offset = ti.Vector([i, j, k])
        for l in ti.static(range(3)):
            weight_0 = w_0[i,3*l+0] * w_0[j,3*l+1] * w_0[k,3*l+2]
            weight_1 = w_1[i,3*l+0] * w_1[j,3*l+1] * w_1[k,3*l+2]
            weight_2 = w_2[i,3*l+0] * w_2[j,3*l+1] * w_2[k,3*l+2]
            weight_3 = w_3[i,3*l+0] * w_3[j,3*l+1] * w_3[k,3*l+2]
            gridv[base_0[:,l] + offset] += weight_0 * bend_force_0[:,l] * dt
            gridv[base_1[:,l] + offset] += weight_1 * bend_force_1[:,l] * dt
            gridv[base_2[:,l] + offset] += weight_2 * bend_force_2[:,l] * dt
            gridv[base_3[:,l] + offset] += weight_3 * bend_force_3[:,l] * dt



TODO 现在用完立刻覆盖 下来重头戏来了 我们要开始做偏移了
for f in facex:

    for l in range(3):
        ef = edgeshared_ownedfaceid[f][l]
        if ef != -1:
            if edgeshared_used[ef] == 0: # 这个边是edgeshared 且从来没有被计算过
                idxvec = cloth_bend.edgeshared_ptclid[ef]
                idx0, idx1, idx2, idx3 = idxvec[0], idxvec[1], idxvec[2], idxvec[3]
                x0, x1, x2, x3 = ptclx[idx0], ptclx[idx1], ptclx[idx2], ptclx[idx3]
                base_0, fx_0, w_0 = ufuncs.B_spline_weight(x0, inv_dx)
                base_1, fx_1, w_1 = ufuncs.B_spline_weight(x1, inv_dx)
                base_2, fx_2, w_2 = ufuncs.B_spline_weight(x2, inv_dx)
                base_3, fx_3, w_3 = ufuncs.B_spline_weight(x3, inv_dx)

                bend_force = cloth_bend.get_cloth_bend_force(ef, (idx0, idx1, idx2, idx3), (x0, x1, x2, x3))

                ti.atomic_add(edgeshared_used[ef], 1)

                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    weight_0 = w_0[i,0] * w_0[j,1] * w_0[k,2]
                    weight_1 = w_1[i,0] * w_1[j,1] * w_1[k,2]
                    weight_2 = w_2[i,0] * w_2[j,1] * w_2[k,2]
                    weight_3 = w_3[i,0] * w_3[j,1] * w_3[k,2]
                    gridv[base_0 + offset] += weight_0 * bend_force[0] * dt
                    gridv[base_1 + offset] += weight_1 * bend_force[1] * dt
                    gridv[base_2 + offset] += weight_2 * bend_force[2] * dt
                    gridv[base_3 + offset] += weight_3 * bend_force[3] * dt

@ti.kernel
def get_edgeshared_ownedfaceid(self, edgeshared_ownedfaceid:ti.template(),
                                    edgeshared_used:ti.template(), ):
    edgeshared_ownedfaceid.fill(-1)

    for e in self.edgeshared:
        es = self.edgeshared[e]
        lface, rface = self.edgeshared_faceid[es][0], self.edgeshared_faceid[es][1]
        lbias = ti.atomic_add(edgeshared_ownedfaceid[lface][3], 1)
        rbias = ti.atomic_add(edgeshared_ownedfaceid[rface][3], 1)
        edgeshared_ownedfaceid[lface][lbias+1] = es
        edgeshared_ownedfaceid[rface][rbias+1] = es

def get_cloth_bend_geometry(self, ):

    self.edgeuniq, self.edgeuniq_faceid, self.edgeuniq_vertid = self.get_edge_flaps(self.vertices)
    num_edgeuniq = self.edgeuniq.shape[0]
    num_edgeshared = self.get_num_edgeshared(self.edgeuniq_faceid)
    self.edgeshared = ti.field(ti.i32, num_edgeshared)
    self.get_edgeshared()

    self.edgeshared_faceid = ti.Vector.field(2, ti.i32, num_edgeuniq) # 每个tripair对应两个face编号
    self.edgeshared_ptclid = ti.Vector.field(4, ti.i32, num_edgeuniq) # 每个triangle pair对应的4个点的index
    self.get_edgeshared_faceid_ptclid()

    self.edgeshared_ownedfaceid = ti.Vector.field(4, ti.i32, self.num_faces)
    self.edgeshared_used = ti.field(ti.i32, num_edgeuniq)
    self.get_edgeshared_ownedfaceid(self.edgeshared_ownedfaceid, self.edgeshared_used)

    self.edgeshared_restphi = ti.field(ti.f32, num_edgeuniq) # 每条边的rest能量
    self.edgeshared_startphi = ti.field(ti.f32, num_edgeuniq) # 每条边的start能量
    self.bending_ka = ti.field(ti.f32, num_edgeuniq)
    self.bending_kb = ti.field(ti.f32, num_edgeuniq)

    return self.edgeshared_ownedfaceid, self.edgeshared_used
"""


