import time
from ros_api import CameraPoseClient
from walker import create_walker, WalkerState

def main():
    walker = create_walker(400)
    walker.run_sequence(WalkerState.WALK, 8)

if __name__ == "__main__":
    main()