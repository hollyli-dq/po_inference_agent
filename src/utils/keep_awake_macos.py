#!/usr/bin/env python3
"""
Keep Awake Script for macOS
This script prevents your Mac from sleeping while long-running tasks execute.
Run this in a separate terminal while your MCMC scripts are running.
"""

import subprocess
import time
import sys
from datetime import datetime

def keep_awake():
    """Keep the Mac awake indefinitely using caffeinate."""
    
    print("=" * 60)
    print("🍎 macOS KEEP AWAKE SYSTEM ACTIVATED")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This script will keep your Mac awake indefinitely.")
    print("Your computer will NOT go to sleep while this is running.")
    print("Press Ctrl+C to stop and allow normal sleep behavior.")
    print("=" * 60)
    print()
    
    try:
        # Use caffeinate to prevent sleep
        # -d: prevent display sleep
        # -i: prevent system idle sleep
        # -s: prevent system sleep (even on AC power)
        process = subprocess.Popen(
            ['caffeinate', '-dims'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Caffeinate started (PID: {process.pid})")
        
        while True:
            # Check if caffeinate is still running
            if process.poll() is not None:
                print("Caffeinate stopped unexpectedly, restarting...")
                process = subprocess.Popen(
                    ['caffeinate', '-dims'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            current_time = datetime.now()
            print(f"[{current_time.strftime('%H:%M:%S')}] ☕ Computer kept awake - Sleep prevention active")
            
            # Status update every 5 minutes
            time.sleep(300)
            
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("STOPPING KEEP AWAKE SYSTEM")
        print("=" * 60)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Terminate caffeinate process
        if process:
            process.terminate()
            process.wait()
            
        print("Normal sleep behavior restored.")
        print("Your Mac can now go to sleep normally.")

def keep_awake_simple():
    """
    Simple one-liner approach - blocks until interrupted.
    Use this if you want the simplest possible solution.
    """
    print("Keeping Mac awake... Press Ctrl+C to stop.")
    subprocess.run(['caffeinate', '-dims'])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--simple':
        keep_awake_simple()
    else:
        keep_awake()







