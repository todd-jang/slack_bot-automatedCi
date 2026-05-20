import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Import custom modules
from models.fsd_net import FSDNet
from dataset.fsd_dataset import FSDDataset
from utils.logger import log_info, log_success, log_warning, log_error, TrainingProgressBar

def parse_args():
    parser = argparse.ArgumentParser(description="E2E FSD Autonomous Driving Training Pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataset")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Force synthetic data generation")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of synthetic samples to generate")
    parser.add_argument("--no-synthetic", action="store_false", dest="synthetic", help="Disable synthetic data (use real data)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    log_info("Initializing E2E FSD Autonomous Driving Training Pipeline...")
    time.sleep(1)
    
    # 1. Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU acceleration
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        
    log_success(f"Using device: {device_name}")
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        log_info(f"Created checkpoints directory at '{args.save_dir}'")

    # 2. Dataset and Loader setup
    # Determine if we have real data directory
    is_synthetic = args.synthetic
    if args.data_dir is None or not os.path.exists(args.data_dir):
        is_synthetic = True
        if args.data_dir:
            log_warning(f"Data directory '{args.data_dir}' not found. Defaulting to synthetic mode.")
            
    dataset = FSDDataset(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        synthetic=is_synthetic
    )
    
    # Split dataset into training and validation (80% train, 20% val)
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, # Set to 0 to prevent multiprocessing overhead on macOS
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    log_success(f"Dataset Split Complete: {train_len} Training samples | {val_len} Validation samples")

    # 3. Model, Optimizer, Loss Setup
    log_info("Building neural network architecture for trajectory prediction...")
    model = FSDNet().to(device)
    time.sleep(1)
    
    # We use MSE loss for regression of trajectory waypoints
    criterion_traj = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    log_info(f"Starting FSD training process. Total Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    
    best_val_loss = float("inf")
    
    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        
        start_time = time.time()
        pbar = TrainingProgressBar(len(train_loader), epoch, args.epochs)
        
        for batch_idx, (cam, lidar, target_traj) in enumerate(train_loader):
            # Move inputs and targets to device
            cam = cam.to(device)
            lidar = lidar.to(device)
            target_traj = target_traj.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_traj = model(cam, lidar)
            
            # Calculate MSE loss for trajectory
            loss_traj = criterion_traj(pred_traj, target_traj)
            
            # Backward pass & Optimize
            loss_traj.backward()
            optimizer.step()
            
            # Track batch metrics
            epoch_train_loss += loss_traj.item()
            
            # Update progress bar
            elapsed = time.time() - start_time
            pbar.update(batch_idx, loss_traj.item(), elapsed)
            
        pbar.finish()
        
        # Calculate training epoch averages
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # 5. Validation Epoch Loop
        model.eval()
        epoch_val_loss = 0.0
        
        # Validation loop for trajectory prediction (no steering/accel metrics needed)
        with torch.no_grad():
            for cam, lidar, target_traj in val_loader:
                cam = cam.to(device)
                lidar = lidar.to(device)
                target_traj = target_traj.to(device)
                
                pred_traj = model(cam, lidar)
                
                loss_traj = criterion_traj(pred_traj, target_traj)
                
                epoch_val_loss += loss_traj.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        # Validation metric is mean squared error (MSE) of predicted waypointss / len(val_loader)
        
        # Clean progress updates for Slack bot consumption
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"Train Loss: {avg_train_loss:.4f} (Trajectory MSE) | Val Loss: {avg_val_loss:.4f} (Trajectory MSE)"
        )
        sys.stdout.flush()
        
        # Save checkpoints
        checkpoint_path = os.path.join(args.save_dir, "fsd_checkpoint_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, checkpoint_path)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.save_dir, "fsd_v1.pth")
            torch.save(model.state_dict(), best_model_path)
            log_success(f"--> Saved best model checkpoint to {best_model_path} (Val Loss: {best_val_loss:.4f})")
            sys.stdout.flush()

    log_success("Training complete! Best model saved to /models/fsd_v1.pth")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_warning("\nTraining interrupted by user. Cleaning up resources...")
        sys.exit(0)
    except Exception as e:
        log_error(f"Fatal error during training: {str(e)}")
        sys.exit(1)
