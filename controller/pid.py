#!/usr/bin/env python3
"""Simple PID controller for steering and throttle.

Provides a class ``SimplePID`` compatible with the demo script
``run_controller.py``. The controller computes a steering command based on
heading error and a throttle command based on distance to the next waypoint.

This is a minimal example for demonstration purposes – a real system would
include more sophisticated vehicle dynamics, feed‑forward terms, and safety
checks.
"""

import math


class SimplePID:
    """PD controller for steering and proportional throttle.

    Parameters
    ----------
    kp_steer: float, default 1.0
        Proportional gain for steering error.
    kd_steer: float, default 0.1
        Derivative gain for steering error.
    kp_throttle: float, default 1.0
        Proportional gain for throttle based on distance to waypoint.
    max_steer: float, default 1.0
        Absolute limit for steering command.
    max_throttle: float, default 1.0
        Absolute limit for throttle command.
    """

    def __init__(self, kp_steer=1.0, kd_steer=0.1, kp_throttle=1.0, max_steer=1.0, max_throttle=1.0):
        self.kp_steer = kp_steer
        self.kd_steer = kd_steer
        self.kp_throttle = kp_throttle
        self.max_steer = max_steer
        self.max_throttle = max_throttle
        self.prev_error = 0.0

    def step(self, position, heading, waypoint):
        """Compute steering and throttle commands.

        Args:
            position (tuple): (x, y) vehicle position.
            heading (float): vehicle heading in radians (0 = +X).
            waypoint (tuple): (x, y) target waypoint.
        Returns:
            (steer, throttle): steering command clipped to ``[-max_steer, max_steer]``
                                 and throttle clipped to ``[0, max_throttle]``.
        """
        # Vector from vehicle to waypoint
        dx = waypoint[0] - position[0]
        dy = waypoint[1] - position[1]
        # Desired heading
        desired = math.atan2(dy, dx)
        # Heading error wrapped to [-pi, pi]
        error = (desired - heading + math.pi) % (2 * math.pi) - math.pi
        # Derivative term
        d_error = error - self.prev_error
        self.prev_error = error
        # PD steering command
        steer = self.kp_steer * error + self.kd_steer * d_error
        steer = max(-self.max_steer, min(self.max_steer, steer))
        # Throttle proportional to distance
        distance = math.hypot(dx, dy)
        throttle = self.kp_throttle * distance
        throttle = max(0.0, min(self.max_throttle, throttle))
        return steer, throttle
