import sys
import time

def log_info(msg):
    """Prints informational message to stdout and flushes."""
    print(f"[*] {msg}")
    sys.stdout.flush()

def log_success(msg):
    """Prints success message to stdout and flushes."""
    print(f"[+] {msg}")
    sys.stdout.flush()

def log_warning(msg):
    """Prints warning message to stdout and flushes."""
    print(f"[!] {msg}")
    sys.stdout.flush()

def log_error(msg):
    """Prints error message to stderr and flushes."""
    print(f"[-] ERROR: {msg}", file=sys.stderr)
    sys.stderr.flush()

class TrainingProgressBar:
    """
    A simple progress bar for tracking batches during an epoch.
    Logs progress incrementally.
    """
    def __init__(self, total_batches, epoch, num_epochs):
        self.total_batches = total_batches
        self.epoch = epoch
        self.num_epochs = num_epochs
        
    def update(self, batch_idx, loss, elapsed_time):
        percent = int(100.0 * (batch_idx + 1) / self.total_batches)
        bar_len = 20
        filled_len = int(bar_len * (batch_idx + 1) // self.total_batches)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        
        # Output status line
        progress_str = (
            f"\rEpoch [{self.epoch}/{self.num_epochs}] |{bar}| {percent}% "
            f"({batch_idx+1}/{self.total_batches}) - Loss: {loss:.4f} - {elapsed_time:.1f}s"
        )
        sys.stdout.write(progress_str)
        sys.stdout.flush()
        
    def finish(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
