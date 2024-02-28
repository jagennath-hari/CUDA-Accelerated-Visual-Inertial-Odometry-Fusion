from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('odom_topic',           default_value='/odom',       description='odom topic'),
        DeclareLaunchArgument('imu_topic',           default_value='/imu',       description='imu topic'),
        Node(
            package = 'cuUKF',
            executable = 'core_node',
            name = 'gpu_filter_node',
            output = 'screen', 
            parameters = [
                {"odom_topic": LaunchConfiguration('odom_topic')},
                {"imu_topic": LaunchConfiguration('imu_topic')}
            ]
        )
    ])
