from setuptools import setup
import os
from glob import glob

package_name = 'fleet_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Member 3 Fleet Manager Node with ML and Safety Integration',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fleet_manager = fleet_manager.fleet_manager_node:main',
            'traffic_predictor = fleet_manager.traffic_prediction_node:main',
        ],
    },
)
