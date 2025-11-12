import numpy as np

def quat_multiply(q1, q0):
    """
    Multiply quaternion q1 by quaternion q0.
    q = [w, x, y, z]
    """
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def quat_apply(q, v):
    """
    Rotate vector v by quaternion q.
    q = [w, x, y, z]
    v = [x, y, z]
    """
    v_quat = np.array([0, v[0], v[1], v[2]])
    q_conj = q * np.array([1, -1, -1, -1])
    return (quat_multiply(quat_multiply(q, v_quat), q_conj))[1:]