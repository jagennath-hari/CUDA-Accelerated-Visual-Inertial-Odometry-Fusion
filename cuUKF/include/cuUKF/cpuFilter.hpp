#ifndef CPUFILTER_HPP_
#define CPUFILTER_HPP_

// ROS includes
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>

// Eigen3 Includes
#if defined __GNUC__ || defined __APPLE__
#include <Eigen/Dense>
#else
#include <eigen3/Eigen/Dense>
#endif

class UKF
{
private:
    // ROS Logger for outsdie node logging
    rclcpp::Logger rosLogger_;

    // 16x1 state vector
    Eigen::VectorXd state_; 

    // 16x16 model covariance matrix
    Eigen::MatrixXd P_;
    void initializeCovarianceMatrix_();

    // 16x16 sensor covariance matrix
    Eigen::MatrixXd R_; // Measurement noise covariance matrix

    // Kalman Gain
    Eigen::MatrixXd K_;

    // Measurement Model Functions
    Eigen::Vector3d prevPosition_;
    
    // Sigma points
    static const int n = 16; // Dimension of the state vector
    Eigen::MatrixXd sigmaPoints_; // Matrix to hold sigma points, size n x (2n + 1)
    Eigen::MatrixXd measurementSigmaPoints_; // Matrix to hold measurment sigma points during tranformation to measuremnt space
    void generateSigmaPoints_(); // gen sigma points

    // UKF parameters
    double lambda_;
    double alpha_;
    double beta_;
    double kappa_;

    // Sigma weights, mean and covaraince
    Eigen::VectorXd weightsMean_;
    Eigen::VectorXd weightsCovariance_;
    Eigen::VectorXd predictedStateMean_; // Predicted state mean
    Eigen::VectorXd predictedMeasurementMean_;
    Eigen::MatrixXd predictedMeasurementCovariance_; // Predicted measurement covariance
    Eigen::MatrixXd Pxy_; // Cross-covariance matrix
    void initializeWeights_();

    // Sigma Process Model functions
    Eigen::Quaterniond updateOrientationSigmaPoint_(const Eigen::Quaterniond& current_orientation, const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& gyro_bias, double dt);
    void updatePositionAndVelocitySigmaPoint_(Eigen::VectorXd& sigmaPoint, const Eigen::Vector3d& accel_meas, double dt);
    void updateBiasesSigmaPoint_(Eigen::VectorXd& sigmaPoint, const Eigen::Vector3d& gyro_error, const Eigen::Vector3d& accel_error);

    // Sigma Measurement Model
    Eigen::VectorXd applyMeasurementModelToSigmaPoint_(const Eigen::VectorXd& sigmaPoint, const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& accel_meas);
    
public:
    UKF();
    ~UKF();

    // Initalize state vector
    void initializeState(const Eigen::VectorXd& initialState, const Eigen::MatrixXd& pose_covariance, const Eigen::MatrixXd& ang_vel_covariance, const Eigen::MatrixXd& accel_covariance);

    // Update prev position
    void updatePrevPose(const Eigen::Vector3d& prevPose);

    // Measurement Model
    void updateNoiseMatrix(const Eigen::MatrixXd& pose_covariance_6x6, const Eigen::MatrixXd& ang_vel_covariance_3x3,const Eigen::MatrixXd& accel_covariance_3x3);
    Eigen::VectorXd updateMeasurementModel(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation, const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& accel_meas, double dt, const Eigen::MatrixXd& pose_covariance_6x6, const Eigen::MatrixXd& ang_vel_covariance_3x3,const Eigen::MatrixXd& accel_covariance_3x3);

    // Prediction Step
    void predictStateMean_();
    void predictStateCovariance_();
    void propagateSigmaPointsThroughProcessModel_(double dt, const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& accel_meas, const Eigen::Vector3d& gyro_error, const Eigen::Vector3d& accel_error);

    // Update Step
    void transformSigmaPointsToMeasurementSpace_(const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& accel_meas);
    void calculatePredictedMeasurementMean_();
    void calculatePredictedMeasurementCovariance_();
    void calculateCrossCovariance_();
    void computeKalmanGain_();
    void updateStateAndCovariance_(const Eigen::VectorXd& actualMeasurement);
};

#endif