# TODO: 解决压缩compactnode在gpu运行方法
# TODO：坐标位置变换有问题

import taichi as ti
import taichi.math as tm
import numpy as np
import time



@ti.data_oriented
class TAICHI_MAKE_LEVELSET:
    def __init__(self, file_vertices_data, file_ppos_data, GRID_SIZE, GRID_BOUND):

        self.GRID_SIZE = GRID_SIZE
        self.GRID_BOUND = GRID_BOUND
        self.MAX_STACK_SIZE = 32
        self.INF_VALUE = 1e9
        self.NOD_VEC_SIZE = 11
        self.CPNOD_VEC_SIZE = 9
        self.IS_LEAF = 1

        self.facenode = ti.Vector.field(3, ti.f32)
        self.num_facenode = ti.field(ti.i32, ())
        self.num_face = ti.field(ti.i32, ())
        self.split_space = ti.Vector.field(3, ti.f32, ())
        self.min_boundary = ti.Vector.field(3, ti.f32, ())
        self.max_boundary = ti.Vector.field(3, ti.f32, ())

        self.sdf_field = ti.field(ti.f32)
        self.sdf_normal = ti.Vector.field(3, ti.f32)
        self.voxel_stack = ti.field(ti.i32)
        self.voxel_map = ti.field(ti.i32)
        ti.root.dense(ti.ijk, [GRID_SIZE,GRID_SIZE,GRID_SIZE]).place(self.sdf_field)
        ti.root.dense(ti.ijk, [GRID_SIZE,GRID_SIZE,GRID_SIZE]).place(self.sdf_normal)
        ti.root.dense(ti.ijk, [GRID_SIZE,GRID_SIZE,self.MAX_STACK_SIZE]).place(self.voxel_stack)
        ti.root.dense(ti.ijk, [GRID_SIZE,GRID_SIZE,GRID_SIZE]).place(self.voxel_map)

        self.load_obj(file_vertices_data, file_ppos_data)

        self.morton_code_s = ti.Vector.field(2, dtype=ti.i32)
        self.morton_code_d = ti.Vector.field(2, dtype=ti.i32)
        ti.root.dense(ti.i, self.num_face[None]).place(self.morton_code_s)
        ti.root.dense(ti.i, self.num_face[None]).place(self.morton_code_d)

        self.face_pot = (self.get_pot_num(self.num_face[None])) << 1 
        self.face_bit = self.get_pot_bit(self.face_pot)

        self.radix_count_zero = ti.field(ti.i32, shape=())
        self.radix_offset = ti.Vector.field(2, ti.i32)
        ti.root.dense(ti.i, self.face_pot).place(self.radix_offset)


        """
        BVH内部节点和叶节点都辈出存在bvh_node里面
        [0]节点类型: 1是leafnode, 0是interiornode
        [1][2]：是左右子树节点编号
        [3]:父节点编号
        [4]:对应faceid
        [5]-[10]:对应包围盒顶点坐标min max
        """
        self.bvh_node = ti.Vector.field(self.NOD_VEC_SIZE, ti.f32) # NOD_VEC_SIZE = 11

        """
        compact_node是经过DFS一维压缩后的bvh_node
        [0]节点类型: 1是leafnode, 0是interiornode
        [1]纪录leafnode偏移位
        [2]纪录interior偏移位
        [3]-[8]:对应包围盒顶点坐标min max
        #                32bit         |32bit       |32bit |96bit  |96bit 
        #compact_node  : is_leaf axis  |face_index  offset  min_v3  max_v3   9
        #                1bit   2bit
        """
        self.compact_node = ti.Vector.field(self.CPNOD_VEC_SIZE, ti.f32) # CPNOD_VEC_SIZE = 9

        self.bvh_done = ti.field(ti.i32, ())

        # node_count表示BVH中的节点总数 叶节点总数num_face[None] 内部节点数量num_face[None]-1
        self.node_count = self.num_face[None]*2-1
        ti.root.dense(ti.i, self.node_count).place(self.bvh_node)
        ti.root.dense(ti.i, self.node_count).place(self.compact_node)

        self.sdf_mindist = ti.field(ti.f32, (GRID_SIZE, GRID_SIZE, GRID_SIZE))
        # 用来表示非INF_VALUE的mindist
        self.voxel_exist_l = ti.Vector.field(3, ti.i32, GRID_SIZE**3)


    def get_pot_num(self, num):
        # 取大于等于numface的最小的2的幂次
        if num <=1 :
            return 1
        else:
            m = 1
            while m<num:
                m = m<<1
            return m>>1

    def get_pot_bit(self, num):
        # facebit表示facepot是2的几次幂 用来控制blellochscan算法中循环次数
        m   = 1
        cnt = 0
        while m<num:
            m = m<<1
            cnt += 1
        return cnt

    def load_obj(self, file_vertices_data, file_ppos_data):
        
        vertices_data_np = np.loadtxt(file_vertices_data, dtype=np.int32, usecols=(1,2,3))
        ppos_data_np = np.loadtxt(file_ppos_data, dtype=np.float32, usecols=(1,2,3))

        num_face_np, num_facenode_np = vertices_data_np.shape[0], 3*vertices_data_np.shape[0]
        ti.root.dense(ti.i, num_facenode_np).place(self.facenode)
        
        self.num_face[None], self.num_facenode[None] = num_face_np, num_facenode_np

        for i,j in ti.ndrange(vertices_data_np.shape[0], 3):
            self.facenode[3*i+j] = ppos_data_np[vertices_data_np[i][j]]

        self.get_minmax_boundary(self.facenode, self.min_boundary, self.max_boundary)

        object_size = (self.max_boundary[None] - self.min_boundary[None]) / (self.GRID_SIZE - 2*self.GRID_BOUND)
        split_space_single = max(object_size)

        self.split_space[None] = tm.vec3(split_space_single)
        self.min_boundary[None] -= self.split_space[None] * self.GRID_BOUND # 最小往外撤GRID_BOUND个
        self.max_boundary[None] = self.min_boundary[None] + self.split_space[None] * self.GRID_SIZE # 最大往外撤GRID_BOUND个

        print("mesh parameters", self.min_boundary, self.max_boundary, 
                self.split_space, self.num_facenode, self.GRID_SIZE)


    @ti.kernel
    def get_minmax_boundary(self, facenode:ti.template(), min_boundary:ti.template(), max_boundary:ti.template()):
        min_boundary[None] = tm.vec3(self.INF_VALUE)
        max_boundary[None] = tm.vec3(-self.INF_VALUE)
        for i in facenode:
            ti.atomic_min(min_boundary[None], facenode[i])
            ti.atomic_max(max_boundary[None], facenode[i])


    #================================================================================
    @ti.func
    def expandBits(self, x):
        '''
        # nvidia blog : https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
        v = ( (v * 0x00010001) & 0xFF0000FF)
        v = ( (v * 0x00000101) & 0x0F00F00F)
        v = ( (v * 0x00000011) & 0xC30C30C3)
        v = ( (v * 0x00000005) & 0x49249249)
        Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
        '''
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x <<  8)) & 0x0300F00F
        x = (x | (x <<  4)) & 0x030C30C3
        x = (x | (x <<  2)) & 0x09249249
        return x

    @ti.func
    def morton3D(self, x, y, z):
        '''
        Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
        '''
        x = tm.min(tm.max(x * 1024.0, 0.0), 1023.0)
        y = tm.min(tm.max(y * 1024.0, 0.0), 1023.0)
        z = tm.min(tm.max(z * 1024.0, 0.0), 1023.0)
        xx = self.expandBits(ti.cast(x, dtype = ti.i32))
        yy = self.expandBits(ti.cast(y, dtype = ti.i32))
        zz = self.expandBits(ti.cast(z, dtype = ti.i32))
        # code = zz | (yy << 1) | (xx << 2)
        code = xx | (yy << 1) | (zz << 2)
        # if code == 0:
        #     print("morton3D",x,y,z)
        return code

    @ti.func
    def gen_morton_tri(self, vertid, posid):
        """
        在0-1区间上获得每个三角形的morton code
        """
        v0 = self.facenode[vertid]
        v1 = self.facenode[vertid+1]
        v2 = self.facenode[vertid+2]
        centre_p = (v1 + v2 + v0) * (1.0 / 3.0)
        norm_p = (centre_p - self.min_boundary[None]) / (self.max_boundary[None] - self.min_boundary[None])
        self.morton_code_s[posid][0] = self.morton3D(norm_p.x, norm_p.y, norm_p.z)
        self.morton_code_s[posid][1] = posid

    #================================================================================

    #================================================================================
    @ti.kernel
    def radix_sort_predicate(self, mask: ti.i32, move: ti.i32):
        for i in self.radix_offset:
            if i < self.num_face[None]:
                self.radix_offset[i][1] = (self.morton_code_s[i][0] & mask) >> move
                self.radix_offset[i][0] = 1 - self.radix_offset[i][1]
                ti.atomic_add(self.radix_count_zero[None], self.radix_offset[i][0]) 
            else:
                self.radix_offset[i][0] = 0
                self.radix_offset[i][1] = 0

    @ti.kernel
    def blelloch_scan_reduce(self, mod: ti.i32):
        for i in self.radix_offset:
            if (i+1)%mod == 0:
                prev_index = i - (mod>>1)
                self.radix_offset[i][0] += self.radix_offset[prev_index][0]
                self.radix_offset[i][1] += self.radix_offset[prev_index][1]

    @ti.kernel
    def blelloch_scan_downsweep(self, mod: ti.i32):
        for i in self.radix_offset:
            if mod == (self.face_pot*2):
                self.radix_offset[self.face_pot-1] = ti.Vector([0,0])
            elif (i+1)%mod == 0:
                prev_index = i - (mod>>1)
                if prev_index >= 0:
                    tmpV   = self.radix_offset[prev_index]
                    self.radix_offset[prev_index] = self.radix_offset[i]
                    self.radix_offset[i] += tmpV

    def blelloch_scan_host(self, mask, move):
        self.radix_sort_predicate(mask, move)
        for i in range(1, self.face_bit+1):
            self.blelloch_scan_reduce(1<<i)
        for i in range(self.face_bit+1, 0, -1):
            self.blelloch_scan_downsweep(1<<i)

    @ti.kernel
    def radix_sort_fill(self, mask: ti.i32, move: ti.i32):
        for i in self.morton_code_s:
            condition = (self.morton_code_s[i][0] & mask ) >> move
            
            if condition == 1:
                offset = self.radix_offset[i][1] + self.radix_count_zero[None]
                self.morton_code_d[offset] = self.morton_code_s[i]
            else:
                offset = self.radix_offset[i][0] 
                self.morton_code_d[offset] = self.morton_code_s[i]

        for i in self.morton_code_s:
            self.morton_code_s[i] = self.morton_code_d[i]
            self.radix_count_zero[None] = 0

    @ti.kernel
    def print_morton_result(self):
        ti.loop_config(serialize=True)
        for i in range(0, self.num_face[None]):
            if i > 0:
                if self.morton_code_s[i][0] < self.morton_code_s[i-1][0]:
                    print(i, self.morton_code_s[i], self.morton_code_s[i-1], "!!!!!!!!!!wrong!!!!!!!!!!!!")
                elif self.morton_code_s[i][0] == self.morton_code_s[i-1][0]:
                    print(i, self.morton_code_s[i], self.morton_code_s[i-1], "~~~~~~equal~~~~~~~~~~~~~")
                else:
                    print(i, self.morton_code_s[i], self.morton_code_s[i-1], "correct morton code")
            else:
                print(i, self.morton_code_s[i], "the very first morton code")  

    #================================================================================


    #================================================================================
    @ti.func
    def common_upper_bits(self, lhs, rhs) :
        """
        用于求两个数的最高公共位
        """
        x = lhs ^ rhs
        ret = 32
        while x > 0:
            x = x>>1
            ret -= 1
            #print(ret, lhs, rhs, x, find, ret)
        #print(x)
        return ret

    @ti.func
    def init_bvh_node(self, index):
        """
        初始化BVH节点,叶子节点index表示在leaf中的位置 非叶子节点index表示在interion中位置 root下表为0
        若此interior节点是parent的leftnode 其下标为其包含最左端leaf节点的下标
        若此interior节点是parent的rightnode 其下标为其包含最右端leaf节点的下标
        """
        #            32bit         | 32bit    | 32bit       | 32bit      | 32bit     |96bit  |96bit 
        #bvh_node  : is_leaf axis  |left_node  right_node   parent_node  face_index   min_v3  max_v3   11
        #             1bit   2bit   
        self.bvh_node[index][0]  = -1.0
        self.bvh_node[index][1]  = -1.0
        self.bvh_node[index][2]  = -1.0
        self.bvh_node[index][3]  = -1.0
        self.bvh_node[index][4]  = -1.0
        self.bvh_node[index][5]  = self.INF_VALUE
        self.bvh_node[index][6]  = self.INF_VALUE
        self.bvh_node[index][7]  = self.INF_VALUE
        self.bvh_node[index][8]  = -self.INF_VALUE
        self.bvh_node[index][9]  = -self.INF_VALUE
        self.bvh_node[index][10] = -self.INF_VALUE

    @ti.func
    def set_node_type(self, index, type):
        """
        第0列的最低位被用来表示节点的类型,以表示该节点是内部节点还是叶节点
        """
        self.bvh_node[index][0] = float(int(self.bvh_node[index][0]) & (0xfffe | type))

    @ti.func
    def set_node_left(self, index, left):
        """
        这个树全部都存在bvhnode里面了 每个index对应的节点都是独一无二的
        第1列纪录leaf节点信息 第2列纪录right节点信息 第3列纪录parent节点信息
        """
        self.bvh_node[index][1]  = float(left)

    @ti.func
    def set_node_right(self, index, right):
        self.bvh_node[index][2]  = float(right)
        
    @ti.func
    def set_node_parent(self, index, parent):
        self.bvh_node[index][3]  = float(parent)

    @ti.func
    def set_node_face(self, index, face):
        """
        设置BVH节点的包含的物体信息 对应第几个face
        """
        self.bvh_node[index][4]  = float(face)

    @ti.func
    def set_node_min_max(self, index, minv,maxv):
        """
        设置BVH节点的包围盒信息
        """
        self.bvh_node[index][5]  = minv[0]
        self.bvh_node[index][6]  = minv[1]
        self.bvh_node[index][7]  = minv[2]
        self.bvh_node[index][8]  = maxv[0]
        self.bvh_node[index][9]  = maxv[1]
        self.bvh_node[index][10] = maxv[2]

    @ti.func
    def init_leaf_node(self, nindex):
        """
        初始化叶节点,即将该节点设置为叶节点并设置其包含的物体信息
        首先使用 set_node_type 函数将节点的类型设置为叶子节点
        """
        self.set_node_type(nindex, self.IS_LEAF)
        face_index = self.morton_code_s[nindex-self.num_face[None]+1][1]
        self.set_node_face(nindex, face_index)
        return face_index

    @ti.func
    def get_tri_min_max(self, vindex, nindex):
        """
        获取一个叶子节点的最小和最大包围盒信息的函数
        其中 vindex 表示三角形的顶点下标, nindex 表示当前叶子节点的下标
        """
        min_v3, max_v3 = tm.vec3(self.INF_VALUE), tm.vec3(-self.INF_VALUE)
        v1 = self.facenode[vindex]
        v2 = self.facenode[vindex+1]
        v3 = self.facenode[vindex+2]
        min_v3 = tm.min(tm.min(tm.min(min_v3,v1),v2),v3)
        max_v3 = tm.max(tm.max(tm.max(max_v3,v1),v2),v3)
        self.set_node_min_max(nindex, min_v3, max_v3)

    @ti.func
    def determine_range(self, idx):
        """
        根据当前节点的Morton码和周围节点的Morton码来确定当前节点所代表的三角形集合的范围。
        它返回一个包含两个元素的整数向量,分别表示左右子节点所代表的三角形范围的起始和结束下标。
        """
        l_r_range = tm.ivec2(0, self.num_face[None]-1)

        """
        如果索引 idx 不是根节点 则检查其左右节点和当前节点的 Morton 码是否相同
        如果相同,那么将左右范围扩展到所有和当前节点 Morton 码相同的三角形
        """
        if idx != 0:
            self_code = self.morton_code_s[idx][0]
            l_code    = self.morton_code_s[idx-1][0]
            r_code    = self.morton_code_s[idx+1][0]

            if  (l_code == self_code) & (r_code == self_code):
                l_r_range[0] = idx
                while idx < self.num_face[None]-1:
                    idx += 1
                    if(idx >= self.num_face[None]-1):
                        break
                    if (self.morton_code_s[idx][0] != self.morton_code_s[idx+1][0]):
                        break
                l_r_range[1] = idx 
            else:
                """
                如果左右节点 Morton 码和当前节点 Morton 码不同 那么需要寻找左右范围
                即所有与当前节点 Morton 码前缀相同的三角形的范围
                首先需要找到左右节点和当前节点 Morton 码的共同前缀长度 L_delta 和 R_delta。
                """
                L_delta = self.common_upper_bits(self_code, l_code)
                R_delta = self.common_upper_bits(self_code, r_code)

                # d是寻找范围的方向
                d = -1
                if R_delta > L_delta:
                    d = 1

                """
                初始化l_max为 2 并且检查距离当前节点 l_max 个三角形的 Morton 码
                与当前节点Morton码的共同前缀长度 delta是否大于共同前缀长度 delta_min。
                """
                delta_min = tm.min(L_delta, R_delta)
                l_max = 2
                delta = -1
                i_tmp = idx + d * l_max

                if ((0 <= i_tmp) & (i_tmp < self.num_face[None])):
                    delta = self.common_upper_bits(self_code, self.morton_code_s[i_tmp][0])

                while delta > delta_min:
                    l_max <<= 1
                    i_tmp = idx + d * l_max
                    delta = -1
                    if ((0 <= i_tmp) & (i_tmp < self.num_face[None])):
                        delta = self.common_upper_bits(self_code, self.morton_code_s[i_tmp][0])

                # 二分查找,寻找左右范围 以l_max的2次幂递增或递减
                l = 0
                t = l_max >> 1
                while(t > 0):
                    i_tmp = idx + (l + t) * d
                    delta = -1
                    if ((0 <= i_tmp) & (i_tmp < self.num_face[None])):
                        delta = self.common_upper_bits(self_code, self.morton_code_s[i_tmp][0])
                    if(delta > delta_min):
                        l += t
                    t >>= 1

                l_r_range[0] = idx
                l_r_range[1] = idx + l * d
                if(d < 0):
                    tmp          = l_r_range[0]
                    l_r_range[0] = l_r_range[1]
                    l_r_range[1] = tmp 

        return l_r_range

    @ti.func
    def find_split(self, first, last):
        """
        当构建BVH树时, 需要对节点进行划分,使得左右子树所包含的三角形的数量尽可能相等,从而达到更好的加速效果。
        find_split函数的作用就是在给定的范围内(从first到last)寻找一个合适的位置将三角形划分到左右两个子树中
        """
        first_code = self.morton_code_s[first][0]
        last_code  = self.morton_code_s[last][0]
        split = first
        if (first_code != last_code):
            delta_node = self.common_upper_bits(first_code, last_code)

            stride = last - first
            while 1:
                stride = (stride + 1) >> 1 # 二分法将当前范围缩小一倍
                middle = split + stride # 得到中间的middle点
                if (middle < last):
                    delta = self.common_upper_bits(first_code, self.morton_code_s[middle][0])
                    if (delta > delta_node):
                        split = middle
                if stride <= 1:
                    break
        return split

    @ti.func
    def build_interior_node(self, nindex):
        self.set_node_type(nindex, 1-self.IS_LEAF)
        l_r_range = self.determine_range(nindex)
        spilt = self.find_split(l_r_range[0], l_r_range[1])

        left_node = spilt
        right_node = spilt + 1

        if tm.min(l_r_range[0], l_r_range[1]) == spilt :
            left_node  += self.num_face[None] - 1
        
        if tm.max(l_r_range[0], l_r_range[1]) == spilt + 1:
            right_node  += self.num_face[None] - 1
        
        if l_r_range[0] == l_r_range[1]:
            print(l_r_range, spilt, left_node, right_node, "wrong")
        # else:
        #     print(l_r_range, spilt,left_node, right_node, "correct")

        self.set_node_left(nindex, left_node)
        self.set_node_right(nindex, right_node)
        self.set_node_parent(left_node, nindex)
        self.set_node_parent(right_node, nindex)

    #================================================================================



    #================================================================================
    @ti.func
    def get_node_min_max(self, index):
        """
        获取BVH节点的包围盒的最小和最大坐标
        """
        return tm.vec3(self.bvh_node[index][5], self.bvh_node[index][6], self.bvh_node[index][7]), \
            tm.vec3(self.bvh_node[index][8], self.bvh_node[index][9], self.bvh_node[index][10])

    @ti.func
    def get_node_has_box(self, index):
        """
        获取BVH节点是否包含有效的包围盒(可以包住的那种)
        """
        return  ((self.bvh_node[index][5] <= self.bvh_node[index][8])
            & (self.bvh_node[index][6] <= self.bvh_node[index][9])
            & (self.bvh_node[index][7] <= self.bvh_node[index][10]))

    @ti.func
    def get_node_child(self, index):
        """
        获取BVH左右子节点编号
        """
        return int(self.bvh_node[index][1]), int(self.bvh_node[index][2])

    @ti.kernel
    def gen_aabb(self):
        """
        获取BVH树节点的包围盒信息 它通过遍历 BVH 树的所有节点,
        对于非叶子节点,如果其左右子节点都已经生成了包围盒信息,
        则可以根据左右子节点的包围盒信息计算出当前节点的包围盒信息
        """
        for i in self.bvh_node:
            if (self.get_node_has_box(i) == 0):
                left_node, right_node = self.get_node_child(i) 
                is_left_rdy = self.get_node_has_box(left_node)
                is_right_rdy = self.get_node_has_box(right_node)
                if (is_left_rdy & is_right_rdy) > 0:
                    l_min,l_max = self.get_node_min_max(left_node) 
                    r_min,r_max = self.get_node_min_max(right_node)
                    self.set_node_min_max(i, tm.min(l_min, r_min), tm.max(l_max, r_max))
                    # 每完成一个包围盒 就加1 当bvhdone的值等于num_face-1时
                    # 说明所有的节点包围盒信息都已经生成了
                    self.bvh_done[None] += 1


    # def flatten_tree(bvh_node_np, compact_node_np, index):
    #     retOffset = self_offset[None]
    #     self_offset[None] += 1

    #     is_leaf = int(bvh_node_np[index, 0]) & 0x0001
    #     left    = int(bvh_node_np[index, 1])
    #     right   = int(bvh_node_np[index, 2])

    #     compact_node_np[retOffset][0] = bvh_node_np[index, 0]
    #     for i in range(6):
    #         compact_node_np[retOffset][3+i] = bvh_node_np[index, 5+i]

    #     if is_leaf != IS_LEAF:
    #         flatten_tree(bvh_node_np, compact_node_np, left)
    #         compact_node_np[retOffset][1] = -1
    #         compact_node_np[retOffset][2] = flatten_tree(bvh_node_np, compact_node_np, right)
    #     else:
    #         compact_node_np[retOffset][1] = bvh_node_np[index, 4]
    #         compact_node_np[retOffset][2] = -1
    #         if node_count == 1:
    #             compact_node_np[retOffset][1] = -1

    #     return retOffset


    def flatten_tree(self, bvh_node_np, compact_node_np, index):
        """
        DFS算法构建compactnode
        """
        self_stack = [(index, None)] # 初始节点为index 没有父节点
        self_offset = 0
        while self_stack:
            # 从stack中弹出来一个index和parentoffset
            index, parent_offset = self_stack.pop()
            ret_offset = self_offset # ret_offset 返回节点的偏移量
            self_offset += 1 # 偏移量计数器

            # 从bvh中读取当当前节点的属性 拿到isleaf 拿到aabb
            is_leaf = int(bvh_node_np[index][0]) & 0x0001
            compact_node_np[ret_offset][0] = bvh_node_np[index][0]
    
            for i in ti.static(range(6)):
                compact_node_np[ret_offset][3+i] = bvh_node_np[index][5+i]

            # 开始压入偏移量
            left = int(bvh_node_np[index][1])
            right = int(bvh_node_np[index][2])

            if is_leaf == self.IS_LEAF:
                compact_node_np[ret_offset][1] = bvh_node_np[index][4]
                compact_node_np[ret_offset][2] = -1
                if self.node_count == 1:
                    compact_node_np[ret_offset][1] = -1
            else:
                self_stack.append((right, ret_offset))
                self_stack.append((left, None))
                compact_node_np[ret_offset][1] = -1

            if parent_offset is not None:
                compact_node_np[parent_offset][2] = ret_offset

    #================================================================================


    #================================================================================
    # TODO：简历bvh的全部过程
    @ti.kernel
    def build_morton_vertex(self, offset:ti.i32):
        """
        为每个三角形生成一个morton code
        """
        for i in range(self.num_face[None]):
            self.gen_morton_tri(3*i+offset, i)

    def radix_sort_host(self, ):
        """
        30位的mortoncode 按照位数用radix sort方法从小到大进行排序 排序后的mortoncode是按照z型排列
        """
        for i in range(30):
            mask = 0x00000001 << i
            self.blelloch_scan_host(mask, i)
            self.radix_sort_fill(mask, i)
        # print_morton_result() # print morton code 可以检查正确排列了

    @ti.kernel
    def build_bvh_vertex(self, offset:ti.int32):
        """
        构建一个bvh的全过程
        """
        for i in self.bvh_node:
            self.init_bvh_node(i)

        for i in self.bvh_node:
            # 找到leafnode
            if i >= self.num_face[None]-1:
                face_index = self.init_leaf_node(i)
                self.get_tri_min_max(3*face_index+offset, i)
            # 找到interiornode
            else:
                self.build_interior_node(i)
        

    def build_bvh_compact(self, ):
        """
        最后需要为整个bvh树的interiornode也建立aabb包围盒信息 并存到bvh_node中
        """
        bvh_done_prev = 0
        self.bvh_done[None] = 0
        while self.bvh_done[None] < self.num_face[None]-1:
            self.gen_aabb()
            if bvh_done_prev == self.bvh_done[None]:
                break
            bvh_done_prev = self.bvh_done[None]
        
        if self.bvh_done[None] != self.num_face[None]-1:
            print("aabb gen error!!!!!!!!!!!!!!!!!!!%d"%self.bvh_done[None])


        """
        构建好bvh树后还需要将它压缩为一维数组 这样在后续遍历过程中
        只需要在一维数组上进行访问和操作即可 而不需要遍历树的结构 
        """
        bvh_node_np = self.bvh_node.to_numpy()
        compact_node_np = self.compact_node.to_numpy()
        self.flatten_tree(bvh_node_np, compact_node_np, 0)
        self.compact_node.from_numpy(compact_node_np)

    # TODO：每次这个都要花掉一半的时间 checkbuild太慢了
    def bvh_setup_vertex(self, offset):
        self.build_morton_vertex(offset)
        self.radix_sort_host()
        self.build_bvh_vertex(offset)

        # start_time = time.time()
        self.build_bvh_compact()
        # end_time = time.time()
        # print("check build is done", end_time-start_time)

    #================================================================================

    #======================================================================
    # TODO：VOXELIZE部分
    # 原来是用barycenter来计算的 返回一个barycenter
    @ti.func
    def intersect_tri(self, origin, direction, face_id):
        """
        https://zhuanlan.zhihu.com/p/476515301
        返回求交的时间t和重心坐标u v
        origin + t*direction = (1-u-v)*A + u*B + v*C
        """
        hit_t = self.INF_VALUE
        u, v = 0.0, 0.0
        vertex_id = 3 * face_id
        v0 = self.facenode[vertex_id + 0]
        v1 = self.facenode[vertex_id + 1]
        v2 = self.facenode[vertex_id + 2]
        E1 = v1 - v0 #edge1
        E2 = v2 - v0 #edge2
        
        P = direction.cross(E2)
        det = E1.dot(P)

        T = tm.vec3(0.0)
        # 如果det<=0,说明direction和三角形所在平面法向量相反
        # 说明origin可能在物体内部 需要把射线反过来
        if (det>0.0):
            T = origin - v0
        else:
            T = v0 - origin
            det = -det

        if (det>0.0):
            u = T.dot(P)
            # 如果u不在这个范围内 表示交点不在三角形内部
            if ((u>0.0) & (u<det)):
                Q = T.cross(E1)
                v = direction.dot(Q)
                if((v>0.0) & (u+v<det)):
                    hit_t = E2.dot(Q)
                    fInvDet = 1.0/det
                    hit_t *= fInvDet
                    u *= fInvDet
                    v *= fInvDet

        return hit_t, u, v

    @ti.func
    def intersect_face(self, origin, direction, face_id):
        """
        求射线和三角形相交的交点 距离
        """
        hit_t = self.INF_VALUE
        hit_pos = tm.vec3(0.0)
        hit_t, u, v = self.intersect_tri(origin, direction, face_id)
        if hit_t < self.INF_VALUE:
            vertid = 3 * face_id
            vert1 = self.facenode[vertid + 0] 
            vert2 = self.facenode[vertid + 1]
            vert3 = self.facenode[vertid + 2]
            hit_pos = (1.0-u-v)*vert1 + u*vert2 + v*vert3  

        return hit_t, hit_pos

    @ti.func
    def get_compact_node_type(self, index):
        """
        表示该节点是内部节点还是叶子节点。在这个函数中,它通过按位与运算获取节点信息中的最后一位,
        如果该位为0则为内部节点 如果为1则为叶子节点。
        """
        return int(self.compact_node[index][0]) & 0x0001 

    @ti.func
    def get_compact_node_face(self, index):
        """
        获取节点所代表的子树中的三角形在三角形数组中的索引。这个信息存储在节点数组的第二列中
        """
        return int(self.compact_node[index][1])

    @ti.func
    def get_compact_node_offset(self, index):
        """
        获取节点在紧凑节点数组中的偏移量。这个信息存储在节点数组的第三列中
        """
        return int(self.compact_node[index][2])

    @ti.func
    def get_compact_node_min_max(self, index):
        """
        获取节点所代表的子树中所有三角形的包围盒的最小坐标和最大坐标。这个信息存储在节点数组的第四到第九列中
        """
        return tm.vec3([self.compact_node[index][3], self.compact_node[index][4], self.compact_node[index][5]]), \
               tm.vec3([self.compact_node[index][6], self.compact_node[index][7], self.compact_node[index][8]])


    @ti.func
    def slabs(self, origin, direction, minv, maxv):
        # most effcient algrithm for ray intersect aabb 
        # en vesrion: https://www.researchgate.net/publication/220494140_An_Efficient_and_Robust_Ray-Box_Intersection_Algorithm
        # cn version: https://zhuanlan.zhihu.com/p/138259656
        """
        判断射线和AABB是否相交 1/0 相交/不相交
        tmin和tmax表示最近和最远的交点
        """
        ret  = 1
        tmin = 0.0
        tmax = self.INF_VALUE
        for i in ti.static(range(3)):
            if abs(direction[i]) < 0.000001:
                if ((origin[i] < minv[i]) | (origin[i] > maxv[i])):
                    ret = 0
            else:
                ood = 1.0 / direction[i] 
                t1 = (minv[i] - origin[i]) * ood 
                t2 = (maxv[i] - origin[i]) * ood
                if(t1 > t2):
                    temp = t1 
                    t1 = t2
                    t2 = temp 
                if(t1 > tmin):
                    tmin = t1
                if(t2 < tmax):
                    tmax = t2 
                if(tmin > tmax) :
                    ret=0
        return ret

    @ti.func
    def closet_hit(self, origin, direction, voxel_stack, i,j, MAX_SIZE):
        """
        寻找射线与BVH中三角形的最近交点和相交面 stack指光线经过BVH的节点堆栈 堆栈坐标ij 最大堆栈MAX_SIZE
        """
        hit_t = self.INF_VALUE
        hit_pos = tm.vec3(0.0)
        hit_face = -1
        voxel_stack[i, j, 0] = 0
        voxel_stack_pos = 0

        while(voxel_stack_pos>=0) & (voxel_stack_pos< MAX_SIZE):
            node_index = voxel_stack[i, j, voxel_stack_pos]
            voxel_stack_pos = voxel_stack_pos - 1

            if self.get_compact_node_type(node_index) == self.IS_LEAF:
                """
                每次从stack中取出一个节点 若该节点为leafnode 就计算射线和该节点包含三角形的交点
                并判断该交点是否是射线与BVH中三角形的最近交点 
                """
                face_index = self.get_compact_node_face(node_index)
                t, pos = self.intersect_face(origin, direction, face_index)
                if (t < hit_t) & (t > 0.0):
                    hit_t       = t
                    hit_pos     = pos
                    hit_face    = face_index
            else:
                """
                如果该节点不是leafnode 然后先拿到该AABB盒的最小最大顶点坐标 
                然后通过slabs判断射线是否和AABB相交 如果相交就把左右子节点压入stack中
                """
                min_v, max_v = self.get_compact_node_min_max(node_index)

                if self.slabs(origin, direction, min_v, max_v) == 1:
                    left_node = node_index + 1
                    right_node = self.get_compact_node_offset(node_index)

                    #push
                    voxel_stack_pos              += 1
                    voxel_stack[i, j, voxel_stack_pos] = left_node
                    voxel_stack_pos              += 1
                    voxel_stack[i, j, voxel_stack_pos] = right_node

        if voxel_stack_pos == MAX_SIZE:
            print("overflow, need larger voxel_stack!!!")

        return hit_t, hit_pos, hit_face

    @ti.kernel
    def voxelize(self, ):
        """
        把模型体素化 现在是从xy面沿着z轴向前扫描 固定住光线方向(0,0,1)不变 从最边角处扩散 
        注意的是一个面可能有好多好多的交点 取决于面的大小
        """
        for i, j in ti.ndrange(self.GRID_SIZE, self.GRID_SIZE):

            ray_origin = self.split_space[None] * tm.vec3(i+0.5,j+0.5,0) + self.min_boundary[None]
            ray_dir = tm.vec3(0,0,1)
            ray_inside = 0

            while 1:
                """
                不断向射线方向发射光线 通过closet_hit判断光线是否与模型相交 计算相交距离 相交位置 相交三角形编号
                """
                hit_t, hit_pos, hit_face = self.closet_hit(ray_origin, ray_dir, self.voxel_stack, i,j, self.MAX_STACK_SIZE)

                if hit_face < 0: # 如果没有相交就退出
                    break

                """
                到这里都是有相交的 计算相交位置的z坐标 计算其voxel的z轴位置 然后将相交位置z坐标和体积元的z坐标比较
                计算在z轴上父该到了哪些体积元 吧他们标记为被物体覆盖(volumemap=1)或没有被覆盖(volumemap=0)
                由于可能出现光线在体积内反复穿凿 因此用变量inside来纪录当前光线是否穿过了物体表面
                """
                zpos = ray_origin.z + hit_t # 拿到hit_pos的z轴坐标
                zhit = (zpos - self.min_boundary[None][2]) / self.split_space[None][2] # 拿到z的voxel坐标
                cur_z = int(tm.floor((ray_origin.z - self.min_boundary[None][2]) / self.split_space[None][2] + 0.5)) # 计算射线起点在z轴上所在格子的位置（整数）
                end_z  = int(tm.min(zhit+0.5, self.GRID_SIZE-1)) # 计算射线在z方向上经过最后一个格子的位置

                while cur_z < end_z:
                    if ray_inside:
                        self.voxel_map[i,j, int(cur_z)] = 1
                    else:
                        self.voxel_map[i,j, int(cur_z)] = 0
                    cur_z += 1

                """
                然后根据相交位置和射线方向计算下一条光线的起点 进入下一轮循环
                最开始时,inside被初始化为0,表示光线从外部进入物体。然后,当光线与三角形相交时,
                将inside的值反转,表示光线穿过了物体表面,进入了物体内部。接着,在光线与下一个三角形相交之前,
                对沿途路径的每个格子进行标记。最后,当光线离开物体时,inside再次被反转为0。
                """
                ray_origin = hit_pos + ti.Vector([0.0, 0.0, 1e-6])
                ray_inside = 1 - ray_inside

    #======================================================================

    #======================================================================
    # 从voxelize搓出来每一帧的sdf

    # sample检测过略掉边界
    @ti.func
    def sample(self, i, j, k):
        """
        采样范围先只在0-GRID_SIZE中
        """
        i = tm.max(0, tm.min(i, self.GRID_SIZE-1))
        j = tm.max(0, tm.min(j, self.GRID_SIZE-1))
        k = tm.max(0, tm.min(k, self.GRID_SIZE-1))
        return self.voxel_map[i, j, k]


    @ti.func
    def sample_sdf(self, i, j, k):
        i = tm.max(0, tm.min(i, self.GRID_SIZE-1))
        j = tm.max(0, tm.min(j, self.GRID_SIZE-1))
        k = tm.max(0, tm.min(k, self.GRID_SIZE-1))
        return self.sdf_field[i,j,k]



    # TODO：到时候就不需要sdf_mindist作为中间变量了 直接把值放到sdf上面就可以了

    @ti.kernel
    def get_sdf_mindist(self, ):
        """
        计算网格上到voxel上的距离 sdf_mindist只纪录到从覆盖voxel里外一层的sdf值
        """
        self.sdf_field.fill(self.INF_VALUE)

        for i, j, k in ti.ndrange(self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE):
            sample_center = self.sample(i, j, k)
            center_exist = (sample_center != 0) # 存在是1 不存在是0
            mindist = self.INF_VALUE
            for r, s, t in ti.static(ti.ndrange((-1,2),(-1,2),(-1,2))):
                offset = tm.ivec3(r+i, s+j, t+k)
                sample_offset = self.sample(offset.x, offset.y, offset.z)
                offset_exist = (sample_offset != 0)
                if(offset_exist != center_exist):
                    dx, dy, dz = r**2, s**2, t**2
                    mindist = tm.min(0.5*tm.sqrt(dx+dy+dz), mindist) 

            self.sdf_mindist[i,j,k] = mindist # 找到距离边界足够近的点


    @ti.kernel
    def get_sdf_field(self, ):
        
        num_ve_l = 0 # 6125

        for i, j, k in ti.ndrange(self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE):
            if self.sdf_mindist[i,j,k] != self.INF_VALUE:
                old_num_ve_l = ti.atomic_add(num_ve_l, 1)
                self.voxel_exist_l[old_num_ve_l] = tm.ivec3(i,j,k)

        for i, j, k in ti.ndrange(self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE):
            if self.sdf_mindist[i,j,k] == self.INF_VALUE:
                min_distance = self.INF_VALUE
                min_pointl = tm.ivec3(self.INF_VALUE)
                for l in range(num_ve_l):
                    vec_ve_l = self.voxel_exist_l[l]
                    vec_ve_diff = vec_ve_l - tm.ivec3([i,j,k])
                    if vec_ve_diff.norm() < min_distance:
                        min_distance = vec_ve_diff.norm()
                        min_pointl = vec_ve_l

                self.sdf_mindist[i,j,k] = min_distance + self.sdf_mindist[min_pointl] 

        sdf_scale = 1.0 / float(self.GRID_SIZE)
        for I in ti.grouped(self.sdf_mindist):
            if self.voxel_map[I] == 0:
                self.sdf_mindist[I] *= sdf_scale
            else:
                self.sdf_mindist[I] *= -sdf_scale

            self.sdf_field[I] = self.sdf_mindist[I]


    @ti.kernel
    def get_sdf_grad(self, ):
        for i, j, k in self.sdf_normal:
            x0 = tm.max(i-1, 0)
            x1 = tm.min(i+1, self.GRID_SIZE-1)
            y0 = tm.max(j-1, 0)
            y1 = tm.min(j+1, self.GRID_SIZE-1)
            z0 = tm.max(k-1, 0)
            z1 = tm.min(k+1, self.GRID_SIZE-1)

            dx = (self.sample_sdf(x1,j,k)-self.sample_sdf(x0,j,k)) * float(self.GRID_SIZE)*0.5 
            dy = (self.sample_sdf(i,y1,k)-self.sample_sdf(i,y0,k)) * float(self.GRID_SIZE)*0.5 
            dz = (self.sample_sdf(i,j,z1)-self.sample_sdf(i,j,z0)) * float(self.GRID_SIZE)*0.5 

            self.sdf_normal[i,j,k] = ti.Vector([dx,dy,dz]).normalized(0.0001)
            if tm.isnan(self.sdf_normal[i,j,k].x):
                print("Iam isnan",x0,y0,z0,dx,dy,dz)


    def get_sdf_field_grad(self, ):
        self.get_sdf_mindist()
        self.get_sdf_field()
        self.get_sdf_grad()


    def make_sdf(self):
        start_time = time.time()
        self.bvh_setup_vertex(0)
        self.voxelize()
        self.get_sdf_field_grad()
        end_time = time.time()
        print("sdf is done", end_time-start_time)
        
        return self.sdf_field, self.sdf_normal 

    #======================================================================



# """
# 如果想要修改make_sdf 直接在这里打开取消掉注释 可以单独运行
# """
# # debug part
# ti.init(arch = ti.gpu, device_memory_fraction=0.7)

# taichi_make_sdf = TAICHI_MAKE_SDF("armadillo_vertices_data.txt", "armadillo_ppos_data.txt", 128, 30)
# taichi_make_sdf.make_sdf()

# my_render_voxels = ti.Vector.field(3, ti.f32, taichi_make_sdf.GRID_SIZE**3)
# my_render_sdf = ti.Vector.field(3, ti.f32, taichi_make_sdf.GRID_SIZE**3)
# my_render_sdfcolor = ti.Vector.field(3, ti.f32, taichi_make_sdf.GRID_SIZE**3)


# @ti.kernel
# def render_voxels():
#     for i,j,k in ti.ndrange(taichi_make_sdf.GRID_SIZE,taichi_make_sdf.GRID_SIZE,taichi_make_sdf.GRID_SIZE):
#         volume_exist = taichi_make_sdf.voxel_map[i,j,k]
#         if volume_exist != 0:
#             # voxelmap只纪录0-128上面有没有值 0-1的绘制范围是我提供的
#             my_render_voxels[taichi_make_sdf.GRID_SIZE*(taichi_make_sdf.GRID_SIZE*i+j)+k] = tm.vec3(
#                 i/taichi_make_sdf.GRID_SIZE, j/taichi_make_sdf.GRID_SIZE, k/taichi_make_sdf.GRID_SIZE)
                


# render_voxels()


# @ti.kernel
# def render_sdf():
#     for i, j, k in ti.ndrange(taichi_make_sdf.GRID_SIZE,taichi_make_sdf.GRID_SIZE,taichi_make_sdf.GRID_SIZE):

#         if taichi_make_sdf.sdf_field[i,j,k] != taichi_make_sdf.INF_VALUE and taichi_make_sdf.sdf_field[i,j,k] <= 0:
#             sdf_index = taichi_make_sdf.GRID_SIZE*(taichi_make_sdf.GRID_SIZE*i+j)+k
#             my_render_sdf[sdf_index] = tm.vec3(i/taichi_make_sdf.GRID_SIZE, j/taichi_make_sdf.GRID_SIZE, k/taichi_make_sdf.GRID_SIZE)
#             sdfcolor = ti.abs(taichi_make_sdf.sdf_field[i,j,k])
#             # print("Iam sdfcolor", sdfcolor)
#             my_render_sdfcolor[sdf_index] = tm.vec3(tm.sin(10*sdfcolor),tm.cos(10*sdfcolor),tm.sin(10*sdfcolor))


# render_sdf()


# def run_ggui():
#     res = (800, 800)
#     window = ti.ui.Window("taichi 3D cloth", res, vsync=True)

#     canvas = window.get_canvas()
#     canvas.set_background_color((1.0, 1.0, 1.0))
#     scene = ti.ui.Scene()

#     camera = ti.ui.Camera()
#     camera.position(0.5, 1.2, 1.95)
#     camera.lookat(0.5, 0.3, 0.5)
#     camera.fov(55)

#     square_bound = ti.Vector.field(3,ti.f32,8)
#     square_bound[0],square_bound[1],square_bound[2] = [0,0,0],[1,0,0],[0,1,0]
#     square_bound[3],square_bound[4],square_bound[5] = [0,0,1],[1,1,0],[0,1,1]
#     square_bound[6],square_bound[7] = [1,0,1],[1,1,1]

#     frame = 0
#     while window.running:
#         camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
#         scene.set_camera(camera)
#         scene.ambient_light((0.1, ) * 3)
#         scene.point_light(pos=(0.5, 10.0, 0.5), color=(0.5, 0.5, 0.5))
#         scene.point_light(pos=(10.0, 10.0, 10.0), color=(0.5, 0.5, 0.5))
#         scene.particles(square_bound, radius=0.02, color=(0.0, 1.0, 1.0))
#         # scene.particles(my_render_voxels, radius=0.002)
#         scene.particles(my_render_sdf, radius=0.002, per_vertex_color=my_render_sdfcolor)
#         # scene.particles(vertex, radius=0.008)

#         canvas.scene(scene)
#         window.show()
#         frame += 1

# if __name__ == '__main__':    
#     run_ggui()




