import os
import csv
import json
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class FSDDataset(Dataset):
    """
    Self-Driving sensor dataset.
    Loads front camera images, LiDAR Bird's Eye View (BEV) arrays,
    and associated steering/acceleration target values.
    
    If `synthetic=True` or no data directory is found, it automatically
    generates synthetic sensor data to allow smooth testing out of the box.
    """
    def __init__(self, data_dir=None, transform=None, num_samples=200, synthetic=True):
        super(FSDDataset, self).__init__()
        self.data_dir = data_dir
        self.synthetic = synthetic
        self.num_samples = num_samples
        
        # Default transforms for camera images if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        self.samples = []
        
        # Check if real data should be loaded
        if not self.synthetic and self.data_dir and os.path.exists(self.data_dir):
            self._load_real_metadata()
        else:
            if not self.synthetic:
                print(f"--> Warning: Data directory '{self.data_dir}' not found or empty. Falling back to Synthetic Mode.")
                self.synthetic = True
            self._generate_synthetic_metadata()

    def _load_real_metadata(self):
        """
        Loads dataset labels and file mappings from a csv file named metadata.csv
        Format: camera_path, lidar_path, steering_angle, acceleration
        """
        metadata_path = os.path.join(self.data_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"--> Warning: metadata.csv not found in {self.data_dir}. Falling back to Synthetic Mode.")
            self.synthetic = True
            self._generate_synthetic_metadata()
            return
            
        try:
            with open(metadata_path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append({
                        "camera_path": os.path.join(self.data_dir, row["camera_path"]),
                        "lidar_path": os.path.join(self.data_dir, row["lidar_path"]),
                        "steering": float(row["steering"]),
                        "acceleration": float(row["acceleration"])
                    })
            self.num_samples = len(self.samples)
            print(f"--> Loaded {self.num_samples} real FSD dataset samples from {self.data_dir}")
        except Exception as e:
            print(f"--> Error loading metadata.csv: {str(e)}. Falling back to Synthetic Mode.")
            self.synthetic = True
            self._generate_synthetic_metadata()

    def _generate_synthetic_metadata(self):
        """
        Generates virtual/synthetic indexes of samples.
        """
        print(f"--> Initializing FSDDataset in SYNTHETIC Mode with {self.num_samples} samples.")
        T = 5  # number of future waypoints
        speed = 5.0  # m/s constant speed for synthetic trajectories
        dt = 0.5    # seconds per waypoint interval
        max_curv = 0.2  # rad per meter, curvature scaling factor
        for i in range(self.num_samples):
            np.random.seed(i)
            # Generate a base steering angle for curvature (-1 to 1)
            steer = float(np.sin(i / 10.0) * 0.5 + np.random.normal(0, 0.05))
            steer = max(-1.0, min(1.0, steer))
            # Generate waypoint coordinates using simple circular arc model
            waypoints = []
            for t_idx in range(T):
                t = (t_idx + 1) * dt
                angle = steer * max_curv * t  # curvature proportional to steering
                x = speed * t * math.cos(angle)
                y = speed * t * math.sin(angle)
                waypoints.append([x, y])
            self.samples.append({
                "id": i,
                "steering": steer,
                "waypoints": torch.tensor(waypoints, dtype=torch.float32)
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.synthetic:
            # Generate synthetic camera image (3, 224, 224)
            np.random.seed(idx)
            img_np = np.zeros((224, 224, 3), dtype=np.uint8)
            img_np[:, :, 1] = 120 + np.random.randint(-10, 10, (224, 224))
            img_np[:, :, 0] = 50
            img_np[:, :, 2] = 50
            for r in range(100, 224):
                width = int(40 + (r - 100) * 0.6)
                center = 112 + int(self.samples[idx]["steering"] * 80 * (r / 224))
                left = max(0, center - width)
                right = min(224, center + width)
                img_np[r, left:right, :] = 80
            img = Image.fromarray(img_np)
            camera_tensor = self.transform(img)
            
            # Synthetic LiDAR BEV map
            lidar_grid = np.zeros((2, 128, 128), dtype=np.float32)
            for r in range(128):
                road_width = int(25 + r * 0.3)
                center = 64 + int(self.samples[idx]["steering"] * 40 * (r / 128))
                left_bound = center - road_width
                right_bound = center + road_width
                if left_bound > 0:
                    lidar_grid[0, r, 0:left_bound] = 1.0
                    lidar_grid[1, r, 0:left_bound] = 0.5
                if right_bound < 128:
                    lidar_grid[0, r, right_bound:128] = 1.0
                    lidar_grid[1, r, right_bound:128] = 0.5
            lidar_tensor = torch.from_numpy(lidar_grid)
            
            # Targets: trajectory waypoints tensor (T,2)
            waypoints = self.samples[idx]["waypoints"]
            return camera_tensor, lidar_tensor, waypoints
            
        else:
            # Load real image
            sample = self.samples[idx]
            camera_img = Image.open(sample["camera_path"]).convert("RGB")
            camera_tensor = self.transform(camera_img)
            lidar_data = np.load(sample["lidar_path"]).astype(np.float32)
            lidar_tensor = torch.from_numpy(lidar_data)
            # Expect real sample to contain precomputed waypoints tensor
            waypoints = torch.tensor(sample["waypoints"], dtype=torch.float32)
            return camera_tensor, lidar_tensor, waypoints

if __name__ == "__main__":
    # Test dataset instantiation & outputs
    dataset = FSDDataset(num_samples=5, synthetic=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    print("Dataset Test:")
    for batch_idx, (cam, lidar, steer, accel) in enumerate(loader):
        print(f"Batch {batch_idx + 1}:")
        print("  Camera tensor shape:", cam.shape)
        print("  LiDAR tensor shape:", lidar.shape)
        print("  Steering shape:", steer.shape, "Values:", steer.squeeze(1).tolist())
        print("  Acceleration shape:", accel.shape, "Values:", accel.squeeze(1).tolist())
