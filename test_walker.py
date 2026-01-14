from walker import WalkerState, create_walker


def main():
    walker = create_walker(400)
    # walker.run_sequence(WalkerState.TURN_RIGHT, 3)
    # walker.run_sequence(WalkerState.TURN_LEFT, 3)
    walker.run_sequence(WalkerState.WALK, 6)


if __name__ == "__main__":
    main()
