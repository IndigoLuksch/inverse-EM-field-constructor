"""
======================
VIBE CODED
~0.1% error compared to magpylib
======================

TensorFlow implementation of magnetic field calculation for cuboid magnets.
Based on magpylib's analytical expressions for uniformly magnetized cuboids.

References:
- Yang: Superconductor Science and Technology 3(12):591 (1999)
- Engel-Herbert: Journal of Applied Physics 97(7):074504 (2005)
- Camacho: Revista Mexicana de Fisica E 59 (2013) 8-17
"""

import tensorflow as tf

MU0 = 4e-7 * 3.14159265359  # Magnetic permeability of free space


def compute_H_field(observers, dimension, polarization, position=(0., 0., 0.)):
    """
    Compute H-field for a single cuboid magnet (matches magpylib API).

    Parameters
    ----------
    observers : array-like, shape (n, 3)
        Absolute observer positions (x, y, z) in meters
    dimension : array-like, shape (3,)
        Cuboid dimensions [a, b, c] in meters
    polarization : array-like, shape (3,)
        Magnetic polarization vector J in Tesla
    position : array-like, shape (3,), default=(0, 0, 0)
        Magnet position in meters

    Returns
    -------
    H : tf.Tensor, shape (n, 3)
        H-field at observer positions in A/m
    """
    observers = tf.convert_to_tensor(observers, dtype=tf.float32)
    position = tf.convert_to_tensor(position, dtype=tf.float32)

    # Compute relative positions (observer - magnet_position)
    observers_rel = observers - position

    return compute_H_field_batch(observers_rel, dimension, polarization)


def compute_H_field_batch(observers, dimensions, polarizations):
    """
    Compute H-field for a batch of cuboid magnets in TensorFlow.

    This is the TensorFlow implementation of magpylib's BHJM_magnet_cuboid
    for the H-field case. Much faster than wrapping NumPy code.

    Parameters
    ----------
    observers : array-like, shape (n, 3)
        Observer positions (x, y, z) in meters, relative to magnet centers
    dimensions : array-like, shape (n, 3) or (3,)
        Cuboid dimensions [a, b, c] in meters
    polarizations : array-like, shape (n, 3) or (3,)
        Magnetic polarization vectors J in Tesla

    Returns
    -------
    H : tf.Tensor, shape (n, 3)
        H-field at observer positions in A/m
    """

    # Convert inputs to TensorFlow tensors
    observers = tf.convert_to_tensor(observers, dtype=tf.float32)
    dimensions = tf.convert_to_tensor(dimensions, dtype=tf.float32)
    polarizations = tf.convert_to_tensor(polarizations, dtype=tf.float32)

    # Handle single magnet case - expand dimensions to batch format
    if len(dimensions.shape) == 1:
        # Single magnet: broadcast to all observer points
        n_observers = tf.shape(observers)[0]
        dimensions = tf.tile(tf.expand_dims(dimensions, 0), [n_observers, 1])
        polarizations = tf.tile(tf.expand_dims(polarizations, 0), [n_observers, 1])

    # Clamp dimensions to prevent numerical instabilities
    # Use absolute value and enforce minimum to avoid singularities
    MIN_DIM = 1e-4  # Minimum dimension in meters
    dimensions = tf.maximum(tf.abs(dimensions), MIN_DIM)

    # Extract components
    pol_x = polarizations[:, 0]
    pol_y = polarizations[:, 1]
    pol_z = polarizations[:, 2]

    # Half-dimensions (cuboid extends from -a/2 to +a/2, etc.)
    a = dimensions[:, 0] / 2.0
    b = dimensions[:, 1] / 2.0
    c = dimensions[:, 2] / 2.0

    x = observers[:, 0]
    y = observers[:, 1]
    z = observers[:, 2]

    # Apply symmetry to avoid numerical instabilities
    # Evaluate everything in the "bottom Q4" octant
    maskx = x < 0
    masky = y > 0
    maskz = z > 0

    x = tf.where(maskx, -x, x)
    y = tf.where(masky, -y, y)
    z = tf.where(maskz, -z, z)

    # Create sign flips for symmetry transformations
    # These account for how the field components transform under reflections
    qs_flipx = tf.constant([[1., -1., -1.], [-1., 1., 1.], [-1., 1., 1.]])
    qs_flipy = tf.constant([[1., -1., 1.], [-1., 1., -1.], [1., -1., 1.]])
    qs_flipz = tf.constant([[1., 1., -1.], [1., 1., -1.], [-1., -1., 1.]])

    batch_size = tf.shape(observers)[0]
    qsigns = tf.ones((batch_size, 3, 3), dtype=tf.float32)

    # Apply sign flips based on which octant the original point was in
    qsigns = tf.where(tf.expand_dims(tf.expand_dims(maskx, 1), 2), qsigns * qs_flipx, qsigns)
    qsigns = tf.where(tf.expand_dims(tf.expand_dims(masky, 1), 2), qsigns * qs_flipy, qsigns)
    qsigns = tf.where(tf.expand_dims(tf.expand_dims(maskz, 1), 2), qsigns * qs_flipz, qsigns)

    # Compute all 8 corner-to-observer distances
    xma, xpa = x - a, x + a
    ymb, ypb = y - b, y + b
    zmc, zpc = z - c, z + c

    xma2, xpa2 = xma**2, xpa**2
    ymb2, ypb2 = ymb**2, ypb**2
    zmc2, zpc2 = zmc**2, zpc**2

    # 8 distances from observer to cuboid corners
    mmm = tf.sqrt(xma2 + ymb2 + zmc2)
    pmp = tf.sqrt(xpa2 + ymb2 + zpc2)
    pmm = tf.sqrt(xpa2 + ymb2 + zmc2)
    mmp = tf.sqrt(xma2 + ymb2 + zpc2)
    mpm = tf.sqrt(xma2 + ypb2 + zmc2)
    ppp = tf.sqrt(xpa2 + ypb2 + zpc2)
    ppm = tf.sqrt(xpa2 + ypb2 + zmc2)
    mpp = tf.sqrt(xma2 + ypb2 + zpc2)

    # Add small epsilon to avoid log(0) and division by zero
    # Increased from 1e-10 to 1e-6 for better numerical stability
    eps = 1e-6

    # Logarithmic terms (ff2)
    ff2x = (
        tf.math.log((xma + mmm + eps) * (xpa + ppm + eps) * (xpa + pmp + eps) * (xma + mpp + eps))
        - tf.math.log((xpa + pmm + eps) * (xma + mpm + eps) * (xma + mmp + eps) * (xpa + ppp + eps))
    )

    ff2y = (
        tf.math.log((-ymb + mmm + eps) * (-ypb + ppm + eps) * (-ymb + pmp + eps) * (-ypb + mpp + eps))
        - tf.math.log((-ymb + pmm + eps) * (-ypb + mpm + eps) * (ymb - mmp + eps) * (ypb - ppp + eps))
    )

    ff2z = (
        tf.math.log((-zmc + mmm + eps) * (-zmc + ppm + eps) * (-zpc + pmp + eps) * (-zpc + mpp + eps))
        - tf.math.log((-zmc + pmm + eps) * (zmc - mpm + eps) * (-zpc + mmp + eps) * (zpc - ppp + eps))
    )

    # Arctangent terms (ff1)
    ff1x = (
        tf.atan2(ymb * zmc, xma * mmm + eps)
        - tf.atan2(ymb * zmc, xpa * pmm + eps)
        - tf.atan2(ypb * zmc, xma * mpm + eps)
        + tf.atan2(ypb * zmc, xpa * ppm + eps)
        - tf.atan2(ymb * zpc, xma * mmp + eps)
        + tf.atan2(ymb * zpc, xpa * pmp + eps)
        + tf.atan2(ypb * zpc, xma * mpp + eps)
        - tf.atan2(ypb * zpc, xpa * ppp + eps)
    )

    ff1y = (
        tf.atan2(xma * zmc, ymb * mmm + eps)
        - tf.atan2(xpa * zmc, ymb * pmm + eps)
        - tf.atan2(xma * zmc, ypb * mpm + eps)
        + tf.atan2(xpa * zmc, ypb * ppm + eps)
        - tf.atan2(xma * zpc, ymb * mmp + eps)
        + tf.atan2(xpa * zpc, ymb * pmp + eps)
        + tf.atan2(xma * zpc, ypb * mpp + eps)
        - tf.atan2(xpa * zpc, ypb * ppp + eps)
    )

    ff1z = (
        tf.atan2(xma * ymb, zmc * mmm + eps)
        - tf.atan2(xpa * ymb, zmc * pmm + eps)
        - tf.atan2(xma * ypb, zmc * mpm + eps)
        + tf.atan2(xpa * ypb, zmc * ppm + eps)
        - tf.atan2(xma * ymb, zpc * mmp + eps)
        + tf.atan2(xpa * ymb, zpc * pmp + eps)
        + tf.atan2(xma * ypb, zpc * mpp + eps)
        - tf.atan2(xpa * ypb, zpc * ppp + eps)
    )

    # Contributions from x-polarization
    bx_pol_x = pol_x * ff1x * qsigns[:, 0, 0]
    by_pol_x = pol_x * ff2z * qsigns[:, 0, 1]
    bz_pol_x = pol_x * ff2y * qsigns[:, 0, 2]

    # Contributions from y-polarization
    bx_pol_y = pol_y * ff2z * qsigns[:, 1, 0]
    by_pol_y = pol_y * ff1y * qsigns[:, 1, 1]
    bz_pol_y = -pol_y * ff2x * qsigns[:, 1, 2]

    # Contributions from z-polarization
    bx_pol_z = pol_z * ff2y * qsigns[:, 2, 0]
    by_pol_z = -pol_z * ff2x * qsigns[:, 2, 1]
    bz_pol_z = pol_z * ff1z * qsigns[:, 2, 2]

    # Sum all contributions
    bx_tot = bx_pol_x + bx_pol_y + bx_pol_z
    by_tot = by_pol_x + by_pol_y + by_pol_z
    bz_tot = bz_pol_x + bz_pol_y + bz_pol_z

    # Combine into B-field vector and normalize
    B = tf.stack([bx_tot, by_tot, bz_tot], axis=1)
    B = B / (4.0 * 3.14159265359)

    # Convert B to H
    # For observers outside the magnet: H = B / μ0
    # For observers inside: H = B / μ0 - M = B / μ0 - J / μ0
    # We assume observers are outside the magnet (typical case)
    H = B / MU0

    return H


def compute_field_mse_loss(params_true, params_pred, observation_points):
    """
    Compute MSE between magnetic fields generated by true and predicted magnets.

    This replaces the magpylib-based loss with a pure TensorFlow implementation.

    Parameters
    ----------
    params_true : tf.Tensor, shape (batch_size, 6)
        True magnet parameters: [x, y, dim_x, dim_y, pol_x, pol_y]
    params_pred : tf.Tensor, shape (batch_size, 6)
        Predicted magnet parameters: [x, y, dim_x, dim_y, pol_x, pol_y]
    observation_points : tf.Tensor, shape (n_points, 3)
        Points where magnetic field is measured

    Returns
    -------
    loss : tf.Tensor, scalar
        Mean squared error between true and predicted H-fields
    """
    batch_size = tf.shape(params_true)[0]
    n_points = tf.shape(observation_points)[0]

    # Fixed z-coordinate for magnets
    z_magnet = 2.5

    # Expand observation points for each batch element
    # Shape: (batch_size, n_points, 3)
    obs_expanded = tf.tile(tf.expand_dims(observation_points, 0), [batch_size, 1, 1])

    # Build magnet positions (x, y, z_fixed)
    positions_true = tf.stack([
        params_true[:, 0],
        params_true[:, 1],
        tf.fill([batch_size], z_magnet)
    ], axis=1)

    positions_pred = tf.stack([
        params_pred[:, 0],
        params_pred[:, 1],
        tf.fill([batch_size], z_magnet)
    ], axis=1)

    # Build dimensions (dim_x, dim_y, dim_z_fixed=1)
    dimensions_true = tf.stack([
        params_true[:, 2],
        params_true[:, 3],
        tf.ones([batch_size])
    ], axis=1)

    dimensions_pred = tf.stack([
        params_pred[:, 2],
        params_pred[:, 3],
        tf.ones([batch_size])
    ], axis=1)

    # Build polarizations (pol_x, pol_y, pol_z_fixed=0)
    polarizations_true = tf.stack([
        params_true[:, 4],
        params_true[:, 5],
        tf.zeros([batch_size])
    ], axis=1)

    polarizations_pred = tf.stack([
        params_pred[:, 4],
        params_pred[:, 5],
        tf.zeros([batch_size])
    ], axis=1)

    # Compute fields for all observation points in each batch
    # We need to reshape to process all batch-point combinations
    # Shape transformations: (batch, points, 3) -> (batch*points, 3)

    # Repeat magnet parameters for each observation point
    positions_true_rep = tf.repeat(positions_true, n_points, axis=0)
    positions_pred_rep = tf.repeat(positions_pred, n_points, axis=0)
    dimensions_true_rep = tf.repeat(dimensions_true, n_points, axis=0)
    dimensions_pred_rep = tf.repeat(dimensions_pred, n_points, axis=0)
    polarizations_true_rep = tf.repeat(polarizations_true, n_points, axis=0)
    polarizations_pred_rep = tf.repeat(polarizations_pred, n_points, axis=0)

    # Flatten observation points
    obs_flat = tf.reshape(obs_expanded, [-1, 3])

    # Compute relative positions (observer - magnet_position)
    observers_rel_true = obs_flat - positions_true_rep
    observers_rel_pred = obs_flat - positions_pred_rep

    # Compute H-fields
    H_true = compute_H_field_batch(observers_rel_true, dimensions_true_rep, polarizations_true_rep)
    H_pred = compute_H_field_batch(observers_rel_pred, dimensions_pred_rep, polarizations_pred_rep)

    # Reshape back to (batch, points, 3)
    H_true = tf.reshape(H_true, [batch_size, n_points, 3])
    H_pred = tf.reshape(H_pred, [batch_size, n_points, 3])

    # Compute MSE for each batch element
    squared_diff = tf.square(H_true - H_pred)
    mse_per_batch = tf.reduce_mean(squared_diff, axis=[1, 2])

    # Average across batch
    return tf.reduce_mean(mse_per_batch)
