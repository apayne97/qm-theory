"""The belt (plate) trick: a belt attached to a wall and to a plate that turns through phi.

The belt is described by the orientation F(s) of its cross-section at each position s in [0, 1] along
its length. The orientation at the wall is the identity, F(0) = 1, and at the plate it is a rotation
through phi about the z axis, F(1) = R_z(phi). Orientations are handled as unit quaternions
q = (w, x, y, z), where q and -q describe the same rotation.

The belt for every phi is an explicit smooth family q(s, phi) that has the following properties
(all checked in test_dirac_belt.py):

* q(0, phi) = 1 and q(1, phi) = (cos(phi/2), 0, 0, sin(phi/2)): the wall and plate conditions.
* q(s, phi + 4*pi) = q(s, phi): the belt returns after a plate rotation of 4*pi, and q(s, 0) = 1 is a
  straight untwisted belt.
* q(s, phi + 2*pi) differs from q(s, phi): after 2*pi the plate looks the same but the belt does not.
  At phi = 2*pi the belt is twisted once about the y axis (the line along which it is laid out).

Construction. The rotations of the plate, phi in [0, 4*pi], form a great circle in the space of unit
quaternions. Every belt q(., phi) is an arc from the identity to a point of this circle, and the arcs
sweep out a disc (a cap of a 2-sphere) bounded by the circle. That a disc exists is the statement that
the 4*pi loop can be shrunk to a point; no such disc exists for the 2*pi path from 1 to -1.
"""

import numpy as np


def plate_quaternion(phi):
    """Quaternion of a rotation through phi about z."""
    phi = np.asarray(phi, dtype=float)
    return np.stack(
        [np.cos(phi / 2), np.zeros_like(phi), np.zeros_like(phi), np.sin(phi / 2)], axis=-1
    )


def belt_quaternions(s, phi):
    """Orientation q(s, phi) of the belt at position s for a plate rotation phi (broadcasts)."""
    s, phi = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(phi, dtype=float))
    # The formula is valid for phi in [0, 4*pi] and the belt is the straight, untwisted one at both ends,
    # so the family is extended to all phi with period 4*pi.
    a = np.mod(phi, 4 * np.pi) / 4
    psi = np.pi * (1 - s)
    w = np.cos(a) ** 2 - np.sin(a) ** 2 * np.cos(psi)
    y = np.sin(a) * np.sin(psi)
    z = np.sin(a) * np.cos(a) * (1 + np.cos(psi))
    return np.stack([w, np.zeros_like(w), y, z], axis=-1)


def quaternion_to_matrix(q):
    """Rotation matrices for unit quaternions of shape (..., 4); returns (..., 3, 3)."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def belt_frames(phi, n=200):
    """Rotation matrices of the belt at n+1 positions s = 0 ... 1; shape (n+1, 3, 3)."""
    s = np.linspace(0, 1, n + 1)
    return quaternion_to_matrix(belt_quaternions(s, phi))


def belt_centerline(frames, length=1.0, axis=1):
    """Centre line of the belt: its tangent at each point is the frame's `axis` direction (y by default).

    Returns an array of shape (n+1, 3) starting at the origin.
    """
    tangent = frames[:, :, axis]
    ds = length / (len(frames) - 1)
    steps = 0.5 * (tangent[1:] + tangent[:-1]) * ds
    return np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])


def ribbon_edges(frames, centerline, width=0.3, axis=0):
    """The two edges of the belt: the centre line displaced by +/- width/2 along the frame's `axis`."""
    offset = 0.5 * width * frames[:, :, axis]
    return centerline + offset, centerline - offset


# ---------------------------------------------------------------------------
# The space of rotations drawn as a ball
# ---------------------------------------------------------------------------


def rotation_vector(q):
    """Rotation vector (axis times angle, angle in [0, pi]) of unit quaternions, shape (..., 3).

    q and -q give the same rotation, so q is flipped to have w >= 0 first. Points of the ball of
    radius pi that lie on its surface and are opposite each other are the same rotation.
    """
    q = np.asarray(q, dtype=float)
    q = np.where(q[..., :1] < 0, -q, q)
    w = np.clip(q[..., 0], -1, 1)
    v = q[..., 1:]
    norm = np.linalg.norm(v, axis=-1)
    angle = 2 * np.arctan2(norm, w)
    safe = np.where(norm > 1e-12, norm, 1.0)
    axis = np.where((norm > 1e-12)[..., None], v / safe[..., None], 0.0)
    return axis * angle[..., None]


def loop_quaternions(kappa, n=400):
    """A loop of rotations that starts and ends at the identity, for kappa in [0, pi/2].

    kappa = pi/2 is the plate rotated through 4*pi (about z). As kappa decreases to 0 the loop is
    shrunk, staying a loop through the identity, until it is the single point 1.
    """
    tau = np.linspace(np.pi, 3 * np.pi, n + 1)
    w = np.cos(kappa) ** 2 - np.sin(kappa) ** 2 * np.cos(tau)
    y = np.cos(kappa) * np.sin(kappa) * (1 + np.cos(tau))
    z = np.sin(kappa) * np.sin(tau)
    return np.stack([w, np.zeros_like(w), y, z], axis=-1)


def single_turn_quaternions(n=400):
    """The plate rotated through 2*pi: a path from the quaternion 1 to -1 (a loop of rotations)."""
    return plate_quaternion(np.linspace(0, 2 * np.pi, n + 1))


def double_turn_quaternions(n=400):
    """The plate rotated through 4*pi: a closed loop of quaternions."""
    return plate_quaternion(np.linspace(0, 4 * np.pi, n + 1))


def surface_crossings(q):
    """Number of times a path of unit quaternions crosses the surface of the rotation ball.

    The crossing happens where w passes through zero (rotation angle pi).
    """
    w = np.asarray(q)[:, 0]
    sign = np.sign(w)
    sign = sign[sign != 0]
    return int(np.sum(sign[1:] != sign[:-1]))


def split_at_surface(vectors, quaternions):
    """Split a path drawn in the ball into pieces, cutting where it jumps to the opposite surface point."""
    w = np.sign(quaternions[:, 0])
    cuts = np.where(w[1:] * w[:-1] < 0)[0] + 1
    return np.split(vectors, cuts)
