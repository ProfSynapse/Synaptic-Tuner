"""
Debug Logger for KTO Training
Helps diagnose where training freezes or hangs
"""

import torch
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingDebugger:
    """Tracks training progress and diagnoses hangs/freezes."""

    def __init__(self, log_file="training_debug.log"):
        self.log_file = log_file
        self.step_times = {}
        self.last_step_time = None

        # Setup file logging
        self.file_handler = logging.FileHandler(log_file, mode='w')
        self.file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.file_handler.setFormatter(formatter)
        logger.addHandler(self.file_handler)
        logger.setLevel(logging.DEBUG)

        logger.info("=" * 80)
        logger.info("Training Debug Logger Started")
        logger.info("=" * 80)

    def log_step_start(self, step: int):
        """Log when a training step starts."""
        self.last_step_time = time.time()
        logger.info(f">>> STEP {step} STARTING")

        # Log memory state
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"    GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def log_step_end(self, step: int, loss: float = None):
        """Log when a training step completes."""
        if self.last_step_time:
            duration = time.time() - self.last_step_time
            self.step_times[step] = duration
            logger.info(f"<<< STEP {step} COMPLETED in {duration:.2f}s" +
                       (f" (loss: {loss:.4f})" if loss else ""))
        else:
            logger.info(f"<<< STEP {step} COMPLETED (no timing)")

    def log_dataloader_start(self, step: int):
        """Log when dataloader starts fetching data."""
        logger.debug(f"    DataLoader: Starting batch fetch for step {step}")

    def log_dataloader_end(self, step: int):
        """Log when dataloader finishes fetching data."""
        logger.debug(f"    DataLoader: Batch ready for step {step}")

    def log_forward_pass_start(self, step: int):
        """Log when forward pass starts."""
        logger.debug(f"    Forward pass: Starting for step {step}")

    def log_forward_pass_end(self, step: int):
        """Log when forward pass ends."""
        logger.debug(f"    Forward pass: Completed for step {step}")

    def log_backward_pass_start(self, step: int):
        """Log when backward pass starts."""
        logger.debug(f"    Backward pass: Starting for step {step}")

    def log_backward_pass_end(self, step: int):
        """Log when backward pass ends."""
        logger.debug(f"    Backward pass: Completed for step {step}")

    def log_optimizer_step_start(self, step: int):
        """Log when optimizer step starts."""
        logger.debug(f"    Optimizer: Starting parameter update for step {step}")

    def log_optimizer_step_end(self, step: int):
        """Log when optimizer step ends."""
        logger.debug(f"    Optimizer: Completed parameter update for step {step}")

    def log_freeze_detection(self, step: int, timeout: int = 300):
        """Called if we suspect training has frozen."""
        logger.error("!" * 80)
        logger.error(f"POTENTIAL FREEZE DETECTED AT STEP {step}")
        logger.error(f"No activity for {timeout} seconds")
        logger.error("!" * 80)

        # Log current state
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.error(f"GPU Memory at freeze: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def log_exception(self, step: int, exception: Exception):
        """Log any exception that occurs."""
        logger.error("!" * 80)
        logger.error(f"EXCEPTION AT STEP {step}")
        logger.error(f"Type: {type(exception).__name__}")
        logger.error(f"Message: {str(exception)}")
        logger.error("!" * 80)

    def get_summary(self):
        """Get a summary of training progress."""
        if not self.step_times:
            return "No steps completed yet"

        avg_time = sum(self.step_times.values()) / len(self.step_times)
        total_steps = len(self.step_times)

        summary = f"""
Training Summary:
  Total steps completed: {total_steps}
  Average step time: {avg_time:.2f}s
  Fastest step: {min(self.step_times.values()):.2f}s
  Slowest step: {max(self.step_times.values()):.2f}s
"""
        return summary

    def close(self):
        """Clean up logger."""
        logger.info("=" * 80)
        logger.info(self.get_summary())
        logger.info("Training Debug Logger Closed")
        logger.info("=" * 80)
        logger.removeHandler(self.file_handler)
        self.file_handler.close()


# Watchdog to detect freezes
class FreezeWatchdog:
    """Monitors training and alerts if it appears to freeze."""

    def __init__(self, timeout: int = 300):
        """
        Args:
            timeout: Seconds of no activity before considering it frozen
        """
        self.timeout = timeout
        self.last_activity = time.time()
        self.current_step = 0
        self.is_frozen = False

    def heartbeat(self, step: int):
        """Call this regularly to indicate training is still alive."""
        self.last_activity = time.time()
        self.current_step = step
        self.is_frozen = False

    def check(self):
        """Check if training appears frozen."""
        elapsed = time.time() - self.last_activity
        if elapsed > self.timeout and not self.is_frozen:
            self.is_frozen = True
            logger.error(f"FREEZE DETECTED: No activity for {elapsed:.0f}s at step {self.current_step}")
            return True
        return False
