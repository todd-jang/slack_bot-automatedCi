import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraBranch(nn.Module):
    """
    Processes front camera images (3, 224, 224) and extracts visual features.
    """
    def __init__(self, embedding_dim=128):
        super(CameraBranch, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=0) # -> 110x110
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)               # -> 55x55
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0) # -> 27x27
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)               # -> 13x13
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # -> 11x11
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)               # -> 5x5
        
        self.fc = nn.Linear(64 * 5 * 5, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc(x)))
        return x

class LiDARBranch(nn.Module):
    """
    Processes Bird's Eye View (BEV) LiDAR grid representations (2, 128, 128).
    Channels could represent (occupancy, height).
    """
    def __init__(self, embedding_dim=128):
        super(LiDARBranch, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=0) # -> 62x62
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)               # -> 31x31
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0) # -> 15x15
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)               # -> 7x7
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # -> 5x5
        # No pool3 since spatial resolution is already small
        
        self.fc = nn.Linear(64 * 5 * 5, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc(x)))
        return x

class FSDNet(nn.Module):
    """
    Full Self-Driving sensor fusion network. Fuses Camera and LiDAR embeddings
    and outputs Steering Angle and Acceleration/Braking.
    """
    def __init__(self, embedding_dim=128):
        super(FSDNet, self).__init__()
        self.camera_branch = CameraBranch(embedding_dim)
        self.lidar_branch = LiDARBranch(embedding_dim)
        
        # Sensor fusion & Decision network
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Trajectory waypoint head (predict T future (x, y) coordinates)
        self.T = 5  # number of future waypoints
        self.trajectory_head = nn.Linear(64, self.T * 2)  # output flat vector of size T*2
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh() # Output ranges [-1.0, 1.0]

    def forward(self, camera_img, lidar_bev):
        # Extract features from each sensor modality
        cam_features = self.camera_branch(camera_img)
        lidar_features = self.lidar_branch(lidar_bev)
        
        # Concatenate sensor features
        fused = torch.cat((cam_features, lidar_features), dim=1)
        
        # Decision making network
        x = self.dropout(self.relu(self.fc1(fused)))
        features = self.relu(self.fc2(x))
        
        # Predict trajectory waypoints
        # Output shape: (batch, T, 2)
        traj_flat = self.trajectory_head(features)  # (batch, T*2)
        traj = traj_flat.view(-1, self.T, 2)  # reshape to (batch, T, 2)
        return traj

if __name__ == "__main__":
    # Quick dimensions check
    model = FSDNet()
    dummy_cam = torch.randn(2, 3, 224, 224)
    dummy_lidar = torch.randn(2, 2, 128, 128)
    
    steer, acc = model(dummy_cam, dummy_lidar)
    print("FSDNet Dimension Test Successful!")
    print("Steering output shape:", steer.shape)
    print("Acceleration output shape:", acc.shape)
