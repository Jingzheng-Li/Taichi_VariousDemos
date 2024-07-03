import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug = True)

pos = ti.Vector.field(3, float, 10)

# 二维叉乘是个数字 三维叉乘是个向量啊？
x = ti.Vector([2., 1., -1.0])
y = ti.Vector([4., 3., 0.0 ])
xy = ti.Matrix.cols([x,y])
a = -10
#print(xy)

outer = -0.5
A = 1 * int(outer>0)

print(A)


