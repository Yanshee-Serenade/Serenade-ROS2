from setuptools import find_packages, setup

package_name = 'serenade_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/chatbot.launch.py',
            'launch/vlm_server.launch.py',
            'launch/walker.launch.py',
            'launch/pointcloud_validator.launch.py',
            'launch/all.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='serenade',
    maintainer_email='subcat2077@gmail.com',
    description='Serenade ROS2 robot control package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chatbot_node=chatbot.main:main',
            'vlm_server_node=server.vlm_ros2_server:main',
            'walker_node=walker.ros2_walker_node:main',
            'pointcloud_validator_node=client.pointcloud_ros2_node:main',
        ],
    },
)
