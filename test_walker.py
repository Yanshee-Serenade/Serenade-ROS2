import time

from walker import WalkSequence, create_walker


def main():
    walker = create_walker(400)
    walker.set_scale(29.6057985)

    # Wait for camera
    time.sleep(1)

    walker.run_sequence(WalkSequence(backward=True))


if __name__ == "__main__":
    main()
