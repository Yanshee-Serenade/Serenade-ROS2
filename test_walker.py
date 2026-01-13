import time
from ros_api import CameraPoseClient
from walker import create_walker, WalkerState

def main():
    walker = create_walker(400)
    walker.run_sequence(WalkerState.TURN_LEFT, 6)
    walker.run_sequence(WalkerState.TURN_RIGHT, 6)

if __name__ == "__main__":
    main()