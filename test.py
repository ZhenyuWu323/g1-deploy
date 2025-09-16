
import numpy as np

import numpy as np

# 绕Z轴旋转-90度
R_z_neg90 = np.array([
    [0,  1,  0],
    [-1, 0,  0],
    [0,  0,  1]
])

# 绕Y轴旋转+90度  
R_y_pos90 = np.array([
    [0,  0,  1],
    [0,  1,  0],
    [-1, 0,  0]
])

# 组合旋转：先Z后Y
combined = R_y_pos90 @ R_z_neg90
print(combined)