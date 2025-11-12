import numpy as np


def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (4,).
        vec: The vector in (x, y, z). Shape is (3,).

    Returns:
        The rotated vector in (x, y, z). Shape is (3,).
    """
    # extract components from quaternion
    w = quat[0]
    xyz = quat[1:]  # (x, y, z) components
    
    # compute cross products
    t = np.cross(xyz, vec) * 2
    return vec + w * t + np.cross(xyz, t)


def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (4,).
        vec: The vector in (x, y, z). Shape is (3,).

    Returns:
        The rotated vector in (x, y, z). Shape is (3,).
    """
    # extract components from quaternion
    w = quat[0]
    xyz = quat[1:]  # (x, y, z) components
    
    # compute cross products
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (4,).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (4,).
    """
    return q * np.array([1, -1, -1, -1])


def quat_inv(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Computes the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (4,).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (4,).
    """
    return quat_conjugate(q) / np.maximum(np.sum(q**2), eps)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (4,).
        q2: The second quaternion in (w, x, y, z). Shape is (4,).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (4,).

    Raises:
        ValueError: Input shapes of q1 and q2 are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    
    # extract components from quaternions
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.array([w, x, y, z])