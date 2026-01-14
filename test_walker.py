import time

from walker import WalkerState, create_walker


def main():
    walker = create_walker(400)
    walker.set_scale(28.5802366)

    # Wait for camera
    time.sleep(1)

    # walker.run_sequence(WalkerState.TURN_RIGHT, 3)
    # walker.run_sequence(WalkerState.TURN_LEFT, 3)
    walker.run_sequence(WalkerState.WALK, 128)


if __name__ == "__main__":
    main()
