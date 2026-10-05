"""Monte Carlo localization: a particle filter over the rover's pose (x, y, yaw) in the map.

Every update moves the particles by the change in odometry (motion model), weights them by how well
the scan fits the map from each particle's pose (sensor model), and resamples when the weights get
uneven. You write the four functions above ParticleFilter; ParticleFilter ties them together.
"""
import math

import numpy as np

from rover_autonomy.geometry import wrap

NUM_PARTICLES = 1000
ALPHAS = (0.02, 0.02, 0.02, 0.02)  # motion noise: turn from turning, turn from driving, drive from driving, drive from turning
SIGMA_HIT = 0.15                   # m: how far a beam endpoint lands from the obstacle it hit
Z_HIT, Z_RAND = 0.9, 0.1           # odds a beam hit a mapped obstacle vs anything else (like an unmapped rock)
MAX_RANGE = 12.0                   # m: the lidar's range
BEAM_STEP = 6                      # use every 6th beam, so 60 of the 360
MIN_MOVE, MIN_TURN = 0.02, 0.02    # m, rad: skip the update while the rover is parked


def sample_motion_model(particles, odom_prev, odom_now, rng):
    """Move every particle by the odometry change, with noise. Returns a new (N, 3) array.

    particles: (N, 3) array of (x, y, yaw) in the map frame.
    odom_prev, odom_now: odometry poses (x, y, yaw) in the odom frame, before and after.
    rng: a numpy Generator; draw noise with rng.normal(0.0, std, N).

    Express the change as rot1 (turn to face where the rover went), trans (drive straight) and rot2
    (turn to the final heading). Give every particle its own noisy copy, with standard deviations
        rot1:  sqrt(a1 * rot1**2 + a2 * trans**2)
        trans: sqrt(a3 * trans**2 + a4 * (rot1**2 + rot2**2))
        rot2:  sqrt(a1 * rot2**2 + a2 * trans**2)
    and replay it from the particle's own heading. Wrap the yaws. Reference: Probabilistic Robotics,
    table 5.6. Watch out when trans is nearly zero (turning in place) and when driving backward.
    """

    # Calculate rot1, trans, and rot2 from odom_prev and odom_now
    x0, y0, yaw0 = odom_prev
    x1, y1, yaw1 = odom_now
    trans = math.sqrt((y1-y0)**2 + (x1-x0)**2)
    direction = np.arctan2(y1-y0, x1-x0)
    rot1 = wrap(direction - yaw0)
    rot2 = wrap(yaw1 - direction)

    # Edge case - rover doesn't move
    if abs(trans) < 0.001: 
        rot1 = 0
        trans = 0
        rot2 = wrap(yaw1 - yaw0)

    # Edge case - driving backward
    if abs(rot1) > math.pi/2:
        trans = -trans
        rot1 = wrap(rot1 + math.pi)
        rot2 = wrap(rot2 - math.pi)

    # Calculate the standard deviations
    a1, a2, a3, a4 = ALPHAS
    rot1_stdev = math.sqrt(a1 * rot1**2 + a2 * trans**2)
    trans_stdev = math.sqrt(a3 * trans**2 + a4 * (rot1**2 + rot2**2))
    rot2_stdev = math.sqrt(a1 * rot2**2 + a2 * trans**2)

    # Add noise to the motion
    N = len(particles)
    noisy_rot1 = rot1 + rng.normal(0.0, rot1_stdev, N)
    noisy_trans = trans + rng.normal(0.0, trans_stdev, N)
    noisy_rot2 = rot2 + rng.normal(0.0, rot2_stdev, N)

    # Apply motion to particles
    new_yaw = particles[:, 2] + noisy_rot1
    new_x = particles[:, 0] + noisy_trans * np.cos(new_yaw)
    new_y = particles[:, 1] + noisy_trans * np.sin(new_yaw)
    new_yaw = wrap(new_yaw + noisy_rot2)
    new_particles = np.column_stack([new_x, new_y, new_yaw])

    return new_particles

def scan_log_likelihood(particles, ranges, angles, grid, laser_offset):
    """How well the scan fits the map from each particle. Returns an (N,) array of log-likelihoods.

    ranges, angles: (B,) arrays of beam lengths and their angles in the laser frame. All are valid hits.
    grid: the map. grid.distance_at(x, y, outside=...) gives the distance to the nearest mapped
        obstacle at world points, for arrays of any shape.
    laser_offset: (x, y, yaw) of the laser in the rover's frame.

    For every particle and beam, find where the beam ends in the map, look up the distance d from
    that endpoint to the nearest obstacle, and score it as
        Z_HIT * N(d; 0, SIGMA_HIT) + Z_RAND / MAX_RANGE
    where N is the Gaussian density. Sum the logs of the scores over the beams. Endpoints off the map
    should get the lowest score, not the highest. Try to do it without Python loops.
    """

    # Get the lidar poses for each particle
    p_x = particles[:, 0]
    p_y = particles[:, 1]
    p_yaw = particles[:, 2]
    l_x, l_y, l_yaw = laser_offset
    lidar_x = p_x + l_x * np.cos(p_yaw) - l_y * np.sin(p_yaw)
    lidar_y = p_y + l_x * np.sin(p_yaw) + l_y * np.cos(p_yaw)
    lidar_yaw = p_yaw + l_yaw
    
    # Get the beam endpoints for every particle and beam
    beam_directions = lidar_yaw[:, None] + angles[None, :]
    x_endpoints = lidar_x[:, None] + ranges[None, :] * np.cos(beam_directions)
    y_endpoints = lidar_y[:, None] + ranges[None, :] * np.sin(beam_directions)

    # On the grid, for every particle and beam, score how close the beam endpoint is to the nearest obstacle
    o_distance = grid.distance_at(x_endpoints, y_endpoints, outside=10000)
    closeness_scores = Z_HIT * np.exp(-o_distance**2 / (2 * SIGMA_HIT**2)) / (SIGMA_HIT * math.sqrt(2 * math.pi)) + Z_RAND / MAX_RANGE

    # Sum the logs of the scores over the beams, for each particle and return total score per particle
    log_scores = np.log(closeness_scores)
    overall_scores = np.sum(log_scores, axis=1)

    return overall_scores

def low_variance_resample(particles, weights, rng):
    """Draw N particles with probability proportional to their weights. Returns a new (N, 3) array.

    Use low-variance (systematic) resampling: one random number r in [0, 1/N), then pointers at
    r, r + 1/N, r + 2/N, ... into the cumulative sum of the weights. Probabilistic Robotics, table 4.4.
    """

    N = len(particles)

    # Low variance resampling
    cumsum = np.cumsum(weights)
    start = rng.uniform(0.0, 1.0 / N)
    pointers = start + np.arange(N) / N

    indices = np.searchsorted(cumsum, pointers, side="right")
    indices = np.minimum(indices, N - 1)

    return particles[indices]

def estimate_pose(particles, weights):
    """The weighted mean pose, as a (3,) array. Average the yaws as angles: the mean of 179 degrees
    and -179 degrees is 180, not 0."""

    # Expected x and y values
    exp_x = np.dot(particles[:, 0], weights)
    exp_y = np.dot(particles[:, 1], weights)

    # Expected yaw value
    yaw_cos = np.dot(np.cos(particles[:, 2]), weights)
    yaw_sin = np.dot(np.sin(particles[:, 2]), weights)
    exp_yaw = np.arctan2(yaw_sin, yaw_cos)

    return np.array([exp_x, exp_y, exp_yaw])

class ParticleFilter:
    """Ties the four functions together. Call step() on every scan."""

    def __init__(self, grid, laser_offset, seed=None):
        self.grid = grid
        self.laser_offset = laser_offset
        self.rng = np.random.default_rng(seed)
        self.particles = None
        self.weights = None
        self.odom = None

    def initialize(self, pose, std=(0.25, 0.25, 0.1)):
        """Spread the particles around a pose."""
        self.particles = np.asarray(pose, dtype=float) + self.rng.normal(0.0, std, (NUM_PARTICLES, 3))
        self.particles[:, 2] = wrap(self.particles[:, 2])
        self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
        self.odom = None

    def step(self, odom, ranges, angles):
        """Update with the odometry pose at the scan's time and the scan.

        Returns (estimate, odom_used): the pose estimate in the map, and the odometry pose it goes with.
        """
        if self.odom is not None:
            moved = math.hypot(odom[0] - self.odom[0], odom[1] - self.odom[1])
            turned = abs(float(wrap(odom[2] - self.odom[2])))
            if moved < MIN_MOVE and turned < MIN_TURN:
                return estimate_pose(self.particles, self.weights), self.odom
            self.particles = sample_motion_model(self.particles, self.odom, odom, self.rng)
        self.odom = np.asarray(odom, dtype=float)

        ranges = np.asarray(ranges, dtype=float)[::BEAM_STEP]
        angles = np.asarray(angles, dtype=float)[::BEAM_STEP]
        hit = np.isfinite(ranges) & (ranges < MAX_RANGE)
        log_w = np.log(self.weights + 1e-300) + scan_log_likelihood(
            self.particles, ranges[hit], angles[hit], self.grid, self.laser_offset)
        w = np.exp(log_w - log_w.max())
        self.weights = w / w.sum()

        if 1.0 / np.sum(self.weights ** 2) < NUM_PARTICLES / 2:
            self.particles = low_variance_resample(self.particles, self.weights, self.rng)
            self.weights = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
        return estimate_pose(self.particles, self.weights), self.odom
