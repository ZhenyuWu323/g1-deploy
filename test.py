
import numpy as np

import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import math


transform = np.array([
                [0,  0,  1],   
                [-1,  0,  0],   
                [0, -1,  0]    
            ])

euler = Rotation.from_matrix(transform).as_euler(seq='XYZ', degrees=True)
print(euler)

flip = np.array([
                [0, -1, 0],
                [-1, 0, 0],
                [0, 0, -1]
            ])
euler_filp = Rotation.from_matrix(flip).as_euler(seq='XYZ', degrees=True)

print(euler_filp)

rot_1 = Rotation.from_euler(angles=[math.pi / 2, - math.pi/2, 0], seq="XYZ")
print(rot_1.as_matrix())