
#ox 616*3 616 vertices positon
#vertices 1770*4 1770 element four vertexid
#indices 1138*3 1138 faces 
#c2e 1770*6 6 edges of an element
#edges 2954*2 total 2954 edges

import numpy as np
import os
import taichi as ti

#ti.init(arch=ti.cpu, debug=True)
ti.init(arch=ti.gpu)

np.set_printoptions(threshold=np.inf)

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def get_rel_path(*segs):
    return os.path.join(SCRIPT_PATH, *segs)


npydata = np.load(get_rel_path('ox_np.npy'))

print(npydata)
print(npydata.shape)



#save vertices_np_quadratic.npy
#=============================================================================
#verticesdata = np.load(get_rel_path('vertices_np.npy'))
#c2edata = np.load(get_rel_path('c2e.npy'))
#newdata = np.zeros((verticesdata.shape[0],10), dtype=int)
#for i in range(1770):
#        newdata[i] = np.append(verticesdata[i], c2edata[i]+616)
#np.save(get_rel_path('vertices_np_quadratic.npy'), newdata)
#=============================================================================

#save ox_np_quadratic.npy
#=============================================================================
#oxdata = np.load(get_rel_path('ox_np.npy'))
#edgesdata = np.load(get_rel_path('edges_np.npy'))
#newdata = np.zeros((oxdata.shape[0]+edgesdata.shape[0],3), dtype=float)
#for i in range(616):
#    newdata[i] = oxdata[i]
#for i in range(2954):
#    newdata[i+616] = 0.5*(oxdata[edgesdata[i][0]]+oxdata[edgesdata[i][1]])
#np.save(get_rel_path('ox_np_quadratic.npy'), newdata)
#=============================================================================
