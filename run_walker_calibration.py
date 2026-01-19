import time

from serenade_ros_api import CameraPoseClient
from serenade_walker import SquatSequence, create_walker


def calibrate_scale():
    """
    Calibrates the scale of the ORB_SLAM3 system by comparing a known
    physical movement (squat 0.05m) against the raw SLAM coordinate change.
    """

    # 1. Initialize Client
    walker = create_walker(1000)
    client = CameraPoseClient(host="localhost", port=21118)

    # We get the generator once so we keep the same TCP connection open
    stream_iterator = client.stream()

    print("[Calibration] Connecting to SLAM stream...")

    def collect_average_y(duration_sec, label):
        """Helper to consume stream for X seconds and average the Y value."""
        y_values = []
        start_time = time.time()

        for topic, data in stream_iterator:
            # Check duration
            if time.time() - start_time > duration_sec:
                break

            # Only process RAW topic for scale calibration
            if topic == "raw":
                # data.position is Position(x, y, z)
                y_values.append(data.position.y)

        if not y_values:
            print(f"[Calibration] Warning: No data received for {label}")
            return 0.0

        avg_y = sum(y_values) / len(y_values)
        print(f"[Calibration] {label}: {avg_y:.7f} (samples: {len(y_values)})")
        return avg_y

    def drain_stream(duration_sec):
        """
        Consumes and discards data.
        Crucial so that after sleeping, we aren't reading old buffered packets.
        """
        start_time = time.time()
        for _ in stream_iterator:
            if time.time() - start_time > duration_sec:
                break

    try:
        # --- Step 1: Measure Initial State ---
        walker.reset()
        print("[Calibration] Waiting 3s for stability...")
        drain_stream(3.0)
        print("[Calibration] Measuring baseline (3s)...")
        y_start = collect_average_y(3.0, "Start Y")

        # --- Step 2: Trigger Squat ---
        print("[Calibration] Robot Squatting...")
        walker.run_sequence(SquatSequence())

        # --- Step 3: Wait for Stability (Drain Buffer) ---
        # We drain the stream for 3 seconds instead of time.sleep(3).
        # This keeps the TCP buffer empty so the next measurement is real-time.
        print("[Calibration] Waiting 3s for stability...")
        drain_stream(3.0)

        # --- Step 4: Measure Squat State ---
        print("[Calibration] Measuring squat position (3s)...")
        y_squat = collect_average_y(3.0, "Squat Y")

        # --- Step 5: Calculate Scale ---
        raw_delta = abs(y_start - y_squat)
        real_delta = 0.05  # 5 cm known physical drop

        if raw_delta < 1e-5:
            print(
                "[Calibration] Error: Raw SLAM y-delta is practically zero. Cannot calibrate."
            )
            scale_factor = 1.0
        else:
            scale_factor = real_delta / raw_delta
            print("-" * 40)
            print(f"Raw Delta:  {raw_delta:.7f}")
            print(f"Real Delta: {real_delta:.7f}")
            print(f"Calculated Scale Factor: {scale_factor:.7f}")
            print("-" * 40)

        # --- Step 6: Revert ---
        print("[Calibration] Reverting to Default...")
        walker.reset()
        return scale_factor

    except Exception as e:
        print(f"[Calibration] Error: {e}")
        return 1

    finally:
        client.close()


if __name__ == "__main__":
    calibrate_scale()
