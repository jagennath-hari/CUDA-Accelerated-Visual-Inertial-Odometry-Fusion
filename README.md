# CUDA-Accelerated-Visual-Inertial-Odometry-Fusion
Harness GPU acceleration for advanced visual odometry and IMU data fusion with our Unscented Kalman Filter (UKF) implementation. Developed with C++ and powered by CUDA, cuBLAS, and cuSOLVER, our system delivers unmatched real-time performance in state and covariance estimation for robotics applications. Integrated with ROS 2 for seamless sensor data management, it ensures high-efficiency and scalable solutions for complex robotic systems, making it ideal for a wide range of autonomous system applications.

## 🏁 Dependencies
1) NVIDIA Driver ([Official Download Link](https://www.nvidia.com/download/index.aspx))
2) CUDA Toolkit ([Official Link](https://developer.nvidia.com/cuda-downloads))
3) ROS 2 Humble ([Official Link](https://docs.ros.org/en/humble/Installation.html))

## ⚙️ Install
1) Clone `https://github.com/jagennath-hari/CUDA-Accelerated-Visual-Inertial-Odometry-Fusion.git`
2) Move `cuUKF` into `ROS2_WORKSPACE`
3) Modify the `CMakeLists.txt` file at `set(CMAKE_CUDA_ARCHITECTURES 89)` and change to your GPU architecture. If you don't know which one you can refer to [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus#compute).
4) `cd ROS2_WORKSPACE` build workspace using `colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc)`

## 📈 Running cuUKF
Launch the node using `ros2 launch cuUKF gpu_filter.launch.py odom_topic:=/odom imu_topic:=/imu`.
The `/odom` should be replaced with your `nav_msgs/Odometry` and the `/imu` should be replaced with your `sensor_msgs/Imu`.

## 💬 ROS 2 Message
The filter odometry gets published as `nav_msgs/Odometry` in the `/cuUKF/filtered_odom` topic.

## 🖼️ RVIZ2 GUI
Launch RVIZ2 using `ros2 run rviz2 rviz2` and subcribe to the `/cuUKF/filered_odom` topic.

<div align="center">
    <img src="assets/Odometry.png" alt="Odometry" width="800"/>
    <p>Odometry</p>
</div>

### Visualize the covariance of the state

<div align="center">
    <img src="assets/Covarince.png" alt="Covariance" width="800"/>
    <p>Covariance</p>
</div>

## ⚠️ Note
1) The fusion does not consider the IMU's orientation only the visual odometry's orientation for the system dynamics and measurements, as raw IMU don't produce orientation without additional filters such as complementary filter and the Madgwick filter.
2) Feel free to change the alpha, beta and kappa values along with the covariance to improve state estimation.
3) The dynamics of the system use simple equations, for the best fusion you may need to change the dynamics.
4) Consider adding augmented sigma points to further increase the robustness.  
5) There is also a CPU version of the UKF for quick development and testing.
