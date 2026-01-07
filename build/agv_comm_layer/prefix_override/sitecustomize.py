import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/debby_anna/ros2_ws/install/agv_comm_layer'
