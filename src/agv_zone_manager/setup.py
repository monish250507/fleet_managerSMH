from setuptools import find_packages, setup

package_name = 'agv_zone_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='debby_anna',
    maintainer_email='debby_anna@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'zone_manager_node = agv_zone_manager.zone_manager_node:main',
        'dummy_odom_publisher = agv_zone_manager.dummy_odom_publisher:main',
    ],
},

)

