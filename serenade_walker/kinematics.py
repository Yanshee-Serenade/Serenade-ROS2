"""
Inverse kinematics solver for robot legs.

This module provides the KinematicsSolver class which interfaces with
the native kinematics library to solve leg inverse kinematics problems.
"""

import ctypes
import math
import os
from typing import List, Optional, Tuple

from ament_index_python.packages import get_package_share_directory


class KinematicsSolver:
    """Inverse kinematics solver for robot legs."""

    def __init__(self):
        """
        Initialize the inverse kinematics solver.

        Raises:
            RuntimeError: If the library cannot be loaded
        """
        lib_path = self._find_library()
        try:
            self.lib = ctypes.CDLL(lib_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load kinematics library {lib_path}: {e}")

        # Set function prototypes
        self.lib.inverse_kinematics.argtypes = [
            ctypes.c_int,  # is_right_leg
            ctypes.c_double,  # target_x
            ctypes.c_double,  # target_y
            ctypes.c_double,  # target_z
            ctypes.c_int,  # grid_density
            ctypes.POINTER(ctypes.c_double),  # angles array
            ctypes.POINTER(ctypes.c_double),  # error
        ]
        self.lib.inverse_kinematics.restype = ctypes.c_int

    @staticmethod
    def _find_library() -> str:
        """
        Find the kinematics library using ROS2 package management.
        
        Returns:
            Path to the libyanshee_kinematics.so library
            
        Raises:
            RuntimeError: If the library cannot be found
        """
        try:
            pkg_dir = get_package_share_directory("yanshee_ik")
            # pkg_dir is typically install/yanshee_ik/share/yanshee_ik
            # we want install/yanshee_ik/lib/libyanshee_kinematics.so
            lib_path = os.path.join(
                os.path.dirname(os.path.dirname(pkg_dir)),
                "lib",
                "libyanshee_kinematics.so"
            )
            if os.path.exists(lib_path):
                return lib_path
        except Exception as e:
            pass
        
        raise RuntimeError(
            "Cannot find libyanshee_kinematics.so. "
            "Make sure the yanshee_ik package is installed and properly built."
        )

    def solve_leg_ik(
        self,
        is_right_leg: bool,
        target_pos: Tuple[float, float, float],
        grid_density: int = 12,
    ) -> Optional[Tuple[List[float], float]]:
        """
        Solve inverse kinematics for a single leg.

        Note: This method does not perform coordinate conversion.
        Coordinate conversion (y-z swap) should be handled by the caller.

        Args:
            is_right_leg: True for right leg, False for left leg
            target_pos: Target position as (x, y, z) tuple
            grid_density: Grid search density

        Returns:
            Tuple of (angles in degrees, error) if successful, None otherwise
        """
        angles = (ctypes.c_double * 5)()
        error = ctypes.c_double()

        x, y, z = target_pos

        result = self.lib.inverse_kinematics(
            ctypes.c_int(1 if is_right_leg else 0),
            ctypes.c_double(x),
            ctypes.c_double(y),
            ctypes.c_double(z),
            ctypes.c_int(grid_density),
            angles,
            ctypes.byref(error),
        )

        if result == 1:
            angles_degrees = [
                round(math.degrees(float(angles[i])), 0) for i in range(5)
            ]
            return angles_degrees, float(error.value)
        else:
            return None
