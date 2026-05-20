#!/usr/bin/env python3
"""Demo script that runs the FSD model on a synthetic sample and
applies a simple PID controller to follow the predicted waypoints.

Steps:
1. Load the best checkpoint (models/fsd_v1.pth).
2. Generate a single synthetic input using ``FSDDataset``.
3. Obtain the predicted waypoint tensor.
4. Starting from the origin, iteratively compute steering and throttle
   commands for each waypoint using the PID controllers.
5. Print the command sequence.

This script showcases how the model output can be turned into low‑level
control actions suitable for a downstream vehicle controller.
"""

import os
import math
import torch
from dataset.fsd_dataset import FSDDataset
from models.fsd_net import FSDNet
from controller.pid import SimplePID


def load_model(checkpoint_path: str = "models/fsd_v1.pth") -> torch.nn.Module:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = FSDNet().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def main():
    # 1) Load model
    checkpoint = os.path.join(os.path.dirname(__file__), "models", "fsd_v1.pth")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}. Run training first.")
    model, device = load_model(checkpoint)
    print(f"Model loaded on device: {device}")

    # 2) Create a single synthetic sample
    dataset = FSDDataset(num_samples=1, synthetic=True)
    cam, lidar, target_waypoints = dataset[0]
    cam = cam.unsqueeze(0).to(device)
    lidar = lidar.unsqueeze(0).to(device)

    # 3) Predict waypoints
    with torch.no_grad():
        pred = model(cam, lidar)
    pred_waypoints = pred.squeeze(0).cpu().numpy()
    print("Predicted waypoints (x, y):")
    for i, wp in enumerate(pred_waypoints, start=1):
        print(f"  WP{i}: ({wp[0]:.2f}, {wp[1]:.2f})")

    # 4) Initialize PID controllers
    pid = SimplePID(kp_steer=1.0, kd_steer=0.1, kp_throttle=1.0, max_steer=1.0, max_throttle=1.0)

    # Vehicle state (starting at origin, heading = 0)
    pos = [0.0, 0.0]
    heading = 0.0  # radians, 0 = +X direction

    print("\nControl commands for each waypoint:")
    for i, wp in enumerate(pred_waypoints, start=1):
        steer, throttle = pid.step(tuple(pos), heading, tuple(wp))
        print(f"  WP{i}: steer={steer:.3f}, throttle={throttle:.3f}")
        # Very simple state update (not a real physics model)
        # Assume fixed small time step dt=0.5s for illustration
        dt = 0.5
        # Update heading based on steering (rough approximation)
        heading += steer * dt
        # Update position assuming forward speed proportional to throttle
        speed = throttle  # m/s approx
        pos[0] += speed * math.cos(heading) * dt
        pos[1] += speed * math.sin(heading) * dt

if __name__ == "__main__":
    main()
