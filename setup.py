from setuptools import find_packages, setup

package_name = 'serenade_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='serenade',
    maintainer_email='subcat2077@gmail.com',
    description='Serenade ROS2 robot control package',
    license='Apache License 2.0',
)
