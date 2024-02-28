#include "coreNode.hpp"

CoreNode::CoreNode() : Node("cuUKF")
{
    // Topics
    this->declare_parameter("odom_topic", "/odom");
    this->declare_parameter("imu_topic", "/imu");

    // Init for UKF
    this->isIntialized_ = false;

    // Initalize filter
    this->kalmanFilter_ = std::make_shared<UKF>(5e-2, 2.0, 0);

    // Subscribe to Msgs
    this->odomDataSub_.subscribe(this, this->get_parameter("odom_topic").as_string());
    this->imuDataSub_.subscribe(this, this->get_parameter("imu_topic").as_string());

    // Initialize the synchronizer
    sync_.reset(new Sync(MySyncPolicy(100), odomDataSub_, imuDataSub_));
    sync_->registerCallback(std::bind(&CoreNode::callback_, this, std::placeholders::_1, std::placeholders::_2));

    // Publisher 
    this->odomPublisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/cuUKF/filtered_odom", 10);
}

CoreNode::~CoreNode()
{
}

void CoreNode::callback_(const nav_msgs::msg::Odometry::ConstSharedPtr& odom, const sensor_msgs::msg::Imu::ConstSharedPtr& imu)
{
    if (!(this->isIntialized_))
    {
        // Set the prevoous time stamp
        this->prevTimestamp_ = odom->header.stamp;
        
        // Initialize state vector
        std::vector<float> state = this->genState_(odom, imu);

        // Move to thrust vector for GPU in filter
        this->kalmanFilter_->initState(state);
        this->kalmanFilter_->initStateCovarianceMatrix(10.0f, 10.0f, 10.0f, 10.0f, 10.0f);
        this->kalmanFilter_->initProcessCovarianceMatrix(10.0f, 10.0f, 10.0f, 10.0f, 10.0f);
        this->kalmanFilter_->updateMeasurementCovarianceMatrix(0.001, 0.001, 0.01f, 0.001, 0.001);
        this->kalmanFilter_->generateSigmaPoints();
        this->kalmanFilter_->initializeWeights();
        
        // Update Prev Pose
        this->kalmanFilter_->updatePrevPose({static_cast<float>(odom->pose.pose.position.x), static_cast<float>(odom->pose.pose.position.y), static_cast<float>(odom->pose.pose.position.z)});

        // Update the flag
        this->isIntialized_ = true;
        return;
    }
    // Get the current timestamp from the odometry message
    rclcpp::Time currentTimestamp = odom->header.stamp;

    // Calculate dt in seconds
    double dt = (currentTimestamp - this->prevTimestamp_).seconds();
    this->prevTimestamp_ = currentTimestamp;  // Update the previous timestamp

    // Start filter
    this->kalmanFilter_->updateFilter(this->genMeasurement_(odom, imu), dt, 0.001, 0.001, 0.01f, 0.001, 0.001);

    // Update Prev Pose
    this->kalmanFilter_->updatePrevPose({static_cast<float>(odom->pose.pose.position.x), static_cast<float>(odom->pose.pose.position.y), static_cast<float>(odom->pose.pose.position.z)});

    std::vector<float> state = this->kalmanFilter_->getState();
    std::vector<float> stateCovarince = this->kalmanFilter_->getStateCovarianceMatrix();
    // Publish Msg
    this->odomPublisher_->publish(this->createOdometryMsg_(state, stateCovarince, odom));
}

std::vector<float> CoreNode::genState_(const nav_msgs::msg::Odometry::ConstSharedPtr& odom, const sensor_msgs::msg::Imu::ConstSharedPtr& imu)
{
    // Initialize state vector
    std::vector<float> state(16, 0.0f); // Initialize all elements to zero

    // Assign position and orientation to state vector
    state[0] = odom->pose.pose.position.x;
    state[1] = odom->pose.pose.position.y;
    state[2] = odom->pose.pose.position.z;
    state[3] = odom->pose.pose.orientation.w;
    state[4] = odom->pose.pose.orientation.x;
    state[5] = odom->pose.pose.orientation.y;
    state[6] = odom->pose.pose.orientation.z;
    state[10] = imu->angular_velocity.x;
    state[11] = imu->angular_velocity.y;
    state[12] = imu->angular_velocity.z;
    state[13] = imu->linear_acceleration.x;
    state[14] = imu->linear_acceleration.y;
    state[15] = imu->linear_acceleration.z;

    return state;
}

std::vector<float> CoreNode::genMeasurement_(const nav_msgs::msg::Odometry::ConstSharedPtr& odom, const sensor_msgs::msg::Imu::ConstSharedPtr& imu)
{
    std::vector<float> measurementVector(16, 0.0f); // Assuming a 16-element state vector

    // Position
    measurementVector[0] = odom->pose.pose.position.x;
    measurementVector[1] = odom->pose.pose.position.y;
    measurementVector[2] = odom->pose.pose.position.z;

    // Orientation (quaternion)
    measurementVector[3] = odom->pose.pose.orientation.w;
    measurementVector[4] = odom->pose.pose.orientation.x;
    measurementVector[5] = odom->pose.pose.orientation.y;
    measurementVector[6] = odom->pose.pose.orientation.z;

    // Angular velocity
    measurementVector[10] = imu->angular_velocity.x;
    measurementVector[11] = imu->angular_velocity.y;
    measurementVector[12] = imu->angular_velocity.z;

    // Linear acceleration
    measurementVector[13] = imu->linear_acceleration.x;
    measurementVector[14] = imu->linear_acceleration.y;
    measurementVector[15] = imu->linear_acceleration.z;

    return measurementVector;
}

nav_msgs::msg::Odometry CoreNode::createOdometryMsg_(const std::vector<float>& state, const std::vector<float>& covariance_matrix, const nav_msgs::msg::Odometry::ConstSharedPtr& original_odom)
{
    nav_msgs::msg::Odometry odom_msg;

    // Assuming the state vector layout is as follows:
    // [x, y, z, qw, qx, qy, qz, vx, vy, vz, ax, ay, az]
    odom_msg.pose.pose.position.x = state[0];
    odom_msg.pose.pose.position.y = state[1];
    odom_msg.pose.pose.position.z = state[2];
    odom_msg.pose.pose.orientation.w = state[3];
    odom_msg.pose.pose.orientation.x = state[4];
    odom_msg.pose.pose.orientation.y = state[5];
    odom_msg.pose.pose.orientation.z = state[6];
    odom_msg.twist.twist.linear.x = state[7];
    odom_msg.twist.twist.linear.y = state[8];
    odom_msg.twist.twist.linear.z = state[9];
    odom_msg.twist.twist.angular.x = state[10];
    odom_msg.twist.twist.angular.y = state[11];
    odom_msg.twist.twist.angular.z = state[12];

    // Copy position covariance (indices 0, 1, 2 for x, y, z)
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) odom_msg.pose.covariance[i * 6 + j] = covariance_matrix[i * 16 + j];

    // Copy orientation covariance (assuming direct mapping from quaternion x, y, z to orientation)
    for (int i = 4, k = 0; i < 7; ++i, ++k) for (int j = 4, l = 0; j < 7; ++j, ++l) odom_msg.pose.covariance[(k + 3) * 6 + (l + 3)] = covariance_matrix[i * 16 + j];

    // Copy velocity covariance (linear velocity: vx, vy, vz)
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) odom_msg.twist.covariance[i * 6 + j] = covariance_matrix[(i + 7) * 16 + (j + 7)];

    // Copy angular velocity covariance (angular velocity: wx, wy, wz)
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) odom_msg.twist.covariance[(i + 3) * 6 + (j + 3)] = covariance_matrix[(i + 10) * 16 + (j + 10)];

    odom_msg.header = original_odom->header;
    odom_msg.child_frame_id = original_odom->child_frame_id;
    // Set the header stamp to the current time
    odom_msg.header.stamp = this->now();

    return odom_msg;
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CoreNode>());
    rclcpp::shutdown();
    
    return 0;
}