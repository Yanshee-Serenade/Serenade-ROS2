"""
Inverse kinematics solver for robot legs.

This module provides the KinematicsSolver class which interfaces with
the native kinematics library to solve leg inverse kinematics problems.
"""

import ctypes
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple


class KinematicsSolver:
    """Inverse kinematics solver for robot legs."""

    def __init__(self):
        """
        Initialize the inverse kinematics solver.

        Args:
            lib_path: Path to the native kinematics library. If None, will search
                     in the ROS2 package installation directory.

        Raises:
            RuntimeError: If the library cannot be loaded
        """
        lib_path = self._find_library()
        try:
            self.lib = ctypes.CDLL(lib_path)
        except Exception as e:
            raise RuntimeError(f"无法加载库 {lib_path}: {e}")

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
        Find the kinematics library in the ROS2 package installation.
        
        Tries multiple locations in order:
        1. ROS2 package install directory (lib/serenade_ros2/)
        2. Current working directory (for development)
        3. System library path
        
        Returns:
            Path to the library
            
        Raises:
            RuntimeError: If the library cannot be found
        """
        lib_name = "libyanshee_kinematics.so"
        
        # Try ROS2 package install directory
        try:
            from ament_index_python.packages import get_package_share_directory
            try:
                pkg_dir = get_package_share_directory("serenade_ros2")
                lib_path = os.path.join(os.path.dirname(pkg_dir), "serenade_ros2", lib_name)
                if os.path.exists(lib_path):
                    return lib_path
            except Exception:
                pass
        except ImportError:
            pass
        
        # Try lib directory next to package
        try:
            from ament_index_python.packages import get_package_share_directory
            try:
                pkg_dir = get_package_share_directory("serenade_ros2")
                # pkg_dir is typically install/serenade_ros2/share/serenade_ros2
                # we want install/serenade_ros2/lib/serenade_ros2/libyanshee_kinematics.so
                lib_path = os.path.join(os.path.dirname(os.path.dirname(pkg_dir)), "lib", "serenade_ros2", lib_name)
                if os.path.exists(lib_path):
                    return lib_path
            except Exception:
                pass
        except ImportError:
            pass
        
        # Try current working directory (development)
        if os.path.exists(lib_name):
            return lib_name
        
        cwd_lib = os.path.join(os.getcwd(), lib_name)
        if os.path.exists(cwd_lib):
            return cwd_lib
        
        # Try package source directory
        try:
            pkg_source = Path(__file__).parent.parent.parent / lib_name
            if pkg_source.exists():
                return str(pkg_source)
        except Exception:
            pass
        
        raise RuntimeError(
            f"Cannot find {lib_name}. Tried ROS2 install directory, "
            f"current directory, and package source directory."
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
