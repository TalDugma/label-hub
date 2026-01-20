
import os
import time
from loguru import logger as guru
from core import init_model

# Configuration
checkpoint = "/Users/taldugma/sam2/checkpoints/sam2.1_hiera_tiny.pt"
cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
log_file = "verify_log.log"

# Clean up old log
if os.path.exists(log_file):
    os.remove(log_file)

guru.add(log_file, level="INFO")

def verify():
    print(f"Initializing model with log_file={log_file}...")
    try:
        # Initialize model
        # We assume 'mps' or 'cpu' availability, defaulting to cpu if mps fails for test stability unless on mac
        device = "cpu"
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            
        model = init_model(checkpoint, cfg, device=device, log_file=log_file)
        
        # Give it a moment to log initialization
        time.sleep(2)
        
        # Check logs
        with open(log_file, "r") as f:
            logs = f.read()
            
        print("\n--- Log Content ---")
        print(logs)
        print("-------------------")
        
        if "Worker: Model loaded" in logs:
            print("\nSUCCESS: Worker log detected!")
        else:
            print("\nFAILURE: Worker log NOT detected.")
            
        if "Initializing SAM2Worker via Proxy" in logs:
             print("SUCCESS: Main process log detected!")
        
        model.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
