"""
Run all three model servers in separate processes.

Starts (configurable via constants):
- VLM Server on port 21122
- DA3 Server on port 21123
- SAM3 Server on port 21124
"""

import multiprocessing
import signal
import sys
import time

# NOTE: Imports are fine here, but heavy CUDA initialization should
# essentially happen inside the target functions, not at the module level.
from server import run_da3_server, run_sam3_server, run_vlm_server

# ==========================================
# CONFIGURATION
# ==========================================
ENABLE_VLM = False
ENABLE_DA3 = True
ENABLE_SAM3 = True
# ==========================================


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🛑 Shutting down all servers...")
    sys.exit(0)


def run_vlm_process():
    """Run VLM server in separate process."""
    try:
        run_vlm_server()
    except KeyboardInterrupt:
        pass


def run_da3_process():
    """Run DA3 server in separate process."""
    try:
        run_da3_server()
    except KeyboardInterrupt:
        pass


def run_sam3_process():
    """Run SAM3 server in separate process."""
    try:
        run_sam3_server()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Context might already be set

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 60)
    print("🚀 Starting Model Servers")
    print("=" * 60)

    processes = []

    # Create processes based on configuration
    if ENABLE_VLM:
        vlm_process = multiprocessing.Process(target=run_vlm_process, name="VLM-Server")
        processes.append(vlm_process)
        print("\n📡 Starting VLM Server (port 21122)...")
        vlm_process.start()
        time.sleep(1)
    else:
        print("\n🚫 VLM Server disabled.")

    if ENABLE_DA3:
        da3_process = multiprocessing.Process(target=run_da3_process, name="DA3-Server")
        processes.append(da3_process)
        print("📡 Starting DA3 Server (port 21123)...")
        da3_process.start()
        time.sleep(1)
    else:
        print("🚫 DA3 Server disabled.")

    if ENABLE_SAM3:
        sam3_process = multiprocessing.Process(
            target=run_sam3_process, name="SAM3-Server"
        )
        processes.append(sam3_process)
        print("📡 Starting SAM3 Server (port 21124)...")
        sam3_process.start()
        time.sleep(1)
    else:
        print("🚫 SAM3 Server disabled.")

    if not processes:
        print("\n⚠️  No servers were enabled. Exiting.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("✅ Selected servers started!")
    print("=" * 60)
    print("\nServers running:")

    for p in processes:
        if p.name == "VLM-Server":
            print(f"  • VLM  (PID {p.pid}): tcp://0.0.0.0:21122")
        elif p.name == "DA3-Server":
            print(f"  • DA3  (PID {p.pid}): tcp://0.0.0.0:21123")
        elif p.name == "SAM3-Server":
            print(f"  • SAM3 (PID {p.pid}): tcp://0.0.0.0:21124")

    print("\nPress Ctrl+C to stop all servers")
    print("=" * 60)

    try:
        # Wait for all started processes
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down all servers...")

        # Terminate all active processes
        for p in processes:
            if p.is_alive():
                p.terminate()

        # Wait for clean shutdown
        for p in processes:
            p.join(timeout=2)

        print("✅ All servers stopped")
