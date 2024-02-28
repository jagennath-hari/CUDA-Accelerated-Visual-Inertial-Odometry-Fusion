#ifndef CPUNODE_HPP_
#define CPUNODE_HPP_

// ROS includes
#include <rclcpp/rclcpp.hpp>

// Message filters
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// Nav and Sensor Msgs includes
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

// Unscneted Kalaman filter include
#include "cpuFilter.hpp"

class CpuNode : public rclcpp::Node
{
private:
    // Message filter subscribers
    message_filters::Subscriber<nav_msgs::msg::Odometry> odomDataSub_;
    message_filters::Subscriber<sensor_msgs::msg::Imu> imuDataSub_;

    // Synchronizer
    typedef message_filters::sync_policies::ApproximateTime<nav_msgs::msg::Odometry, sensor_msgs::msg::Imu> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;

    // Combined callback
    void callback_(const nav_msgs::msg::Odometry::ConstSharedPtr& odom, const sensor_msgs::msg::Imu::ConstSharedPtr& imu);

    // UKF Object
    std::shared_ptr<UKF> kalmanFilter_;

    // Init for UKF
    bool isIntialized_;

    // Member to store the previous timestamp
    rclcpp::Time prevTimestamp_;

    // ROS msg to Eigen conversions
    Eigen::MatrixXd extractPoseCovariance_(const nav_msgs::msg::Odometry& odom);
    Eigen::MatrixXd extractAngularVelocityCovariance_(const sensor_msgs::msg::Imu& imu);
    Eigen::MatrixXd extractLinearAccelerationCovariance_(const sensor_msgs::msg::Imu& imu);

public:
    CpuNode();
    ~CpuNode();
};

#endif
