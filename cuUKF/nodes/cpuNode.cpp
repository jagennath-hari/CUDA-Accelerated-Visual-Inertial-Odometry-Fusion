#include "cpuNode.hpp"

CpuNode::CpuNode() : Node("cuUKF")
{
    // Init for UKF
    this->isIntialized_ = false;

    // Initialize previous timestamp with current ROS time
    this->prevTimestamp_ = this->now();

    // Initalize filter
    this->kalmanFilter_ = std::make_shared<UKF>();

    // Subscribe to Msgs
    this->odomDataSub_.subscribe(this, "/zed2i/zed_node/odom");
    this->imuDataSub_.subscribe(this, "/zed2i/zed_node/imu/data_raw");

    // Initialize the synchronizer
    sync_.reset(new Sync(MySyncPolicy(10), odomDataSub_, imuDataSub_));
    sync_->registerCallback(std::bind(&CpuNode::callback_, this, std::placeholders::_1, std::placeholders::_2));
}

CpuNode::~CpuNode()
{
}

Eigen::MatrixXd CpuNode::extractPoseCovariance_(const nav_msgs::msg::Odometry& odom)
{
    // The pose covariance matrix in Odometry is 6x6, stored as a flat array of 36 elements
    const int matrix_size = 6;

    // Use Eigen's Map to create a 6x6 matrix view of the 1D array without copying the data
    Eigen::Map<const Eigen::Matrix<double, matrix_size, matrix_size, Eigen::RowMajor>> covariance_matrix(odom.pose.covariance.data());

    return covariance_matrix;
}

Eigen::MatrixXd CpuNode::extractAngularVelocityCovariance_(const sensor_msgs::msg::Imu& imu)
{
    const int matrix_size = 3; // The angular velocity covariance matrix in Imu is 3x3

    Eigen::Map<const Eigen::Matrix<double, matrix_size, matrix_size, Eigen::RowMajor>> covariance_matrix(imu.angular_velocity_covariance.data());

    return covariance_matrix;
}

Eigen::MatrixXd CpuNode::extractLinearAccelerationCovariance_(const sensor_msgs::msg::Imu& imu)
{
    const int matrix_size = 3; // The linear acceleration covariance matrix in Imu is 3x3

    Eigen::Map<const Eigen::Matrix<double, matrix_size, matrix_size, Eigen::RowMajor>> covariance_matrix(imu.linear_acceleration_covariance.data());

    return covariance_matrix;
}

void CpuNode::callback_(const nav_msgs::msg::Odometry::ConstSharedPtr& odom, const sensor_msgs::msg::Imu::ConstSharedPtr& imu)
{
    // Get the current timestamp from the odometry message
    rclcpp::Time currentTimestamp = odom->header.stamp;

    // Calculate dt in seconds
    double dt = (currentTimestamp - this->prevTimestamp_).seconds();
    this->prevTimestamp_ = currentTimestamp;  // Update the previous timestamp

    if (!(this->isIntialized_)) 
    {
        // Initialize the state vector with the first readings
        Eigen::Vector3d initial_position(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);
        Eigen::Quaterniond initial_orientation(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
        // Assuming zero initial velocity and biases
        Eigen::VectorXd initial_state(16);
        initial_state << initial_position, initial_orientation.coeffs(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

        this->kalmanFilter_->initializeState(initial_state, this->extractPoseCovariance_(*odom), this->extractAngularVelocityCovariance_(*imu), this->extractLinearAccelerationCovariance_(*imu));
        this->kalmanFilter_->updatePrevPose(initial_position);
        this->isIntialized_ = true;
        return;
    }

    // Extract position and orientation from the odometry message
    Eigen::Vector3d position(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);
    Eigen::Quaterniond orientation(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);

    // Extract angular velocity and linear acceleration from the IMU message
    Eigen::Vector3d angular_velocity(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    Eigen::Vector3d linear_acceleration(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);

    Eigen::Vector3d gyro_error(0.001, 0.001, 0.001); // rad/s
    Eigen::Vector3d accel_error(0.01, 0.01, 0.01); // m/s^2

    // Propagate each sigma point through the process model
    this->kalmanFilter_->propagateSigmaPointsThroughProcessModel_(dt, angular_velocity, linear_acceleration, gyro_error, accel_error);

    // Perform the UKF Prediction Step
    this->kalmanFilter_->predictStateMean_();
    this->kalmanFilter_->predictStateCovariance_();

    // Transform the propagated sigma points into the measurement space
    this->kalmanFilter_->transformSigmaPointsToMeasurementSpace_(angular_velocity, linear_acceleration);
    
    // Calculate the predicted measurement mean
    this->kalmanFilter_->calculatePredictedMeasurementMean_();
    
    // Calculate the predicted measurement covariance
    this->kalmanFilter_->calculatePredictedMeasurementCovariance_();
    
    // Calculate the cross covariance between the state and the measurements
    this->kalmanFilter_->calculateCrossCovariance_();
    
    // Compute the Kalman Gain
    this->kalmanFilter_->computeKalmanGain_();
    
    // Prepare the actual measurement vector from the current odometry and IMU data
    Eigen::VectorXd actualMeasurement = this->kalmanFilter_->updateMeasurementModel(position, orientation, angular_velocity, linear_acceleration, dt, this->extractPoseCovariance_(*odom), this->extractAngularVelocityCovariance_(*imu), this->extractLinearAccelerationCovariance_(*imu));
    
    // Update the state and covariance of the UKF with the actual measurement
    this->kalmanFilter_->updateStateAndCovariance_(actualMeasurement);
    
    // Update the previous position for velocity calculation in the next cycle
    this->kalmanFilter_->updatePrevPose(position);
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CpuNode>());
    rclcpp::shutdown();
    
    return 0;
}