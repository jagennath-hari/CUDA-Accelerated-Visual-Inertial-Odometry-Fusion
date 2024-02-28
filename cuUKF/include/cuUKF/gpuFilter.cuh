#ifndef GPUFILTER_CUH_
#define GPUFILTER_CUH_

// Include
#include <iostream>

// ROS includes
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>

// Nav and Sensor Msgs includes
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

// CUDA includes
#include <cuda_runtime.h>

// cuBLAS for linear algebra operations
#include <cublas_v2.h>

// cuSOLVER for Cholesky decomposition
#include <cusolverDn.h>

// Thrust for device based vector operations
#include <thrust/device_vector.h>

class UKF
{
private:
    // ROS Logger for outsdie node logging
    rclcpp::Logger rosLogger_;  

    // CUDA handles
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
    
    // UKF parameters
    static const int n = 16; // Dimension of the state vector
    static const int sigmaPointCount = 2 * 16 + 1;
    float lambda_;
    float alpha_;
    float beta_;
    float kappa_;  

    // Stores the previous position (x, y, z)
    thrust::device_vector<float> prevPose_;
    // State vector using thrust
    thrust::device_vector<float> state_;
    // State Covariance matrix, linearized using thrust
    thrust::device_vector<float> P_;
    // Process Covariance matrix, linearized using thrust
    thrust::device_vector<float> Q_;
    // Measurement Covariance matrix, linearized using thrust
    thrust::device_vector<float> R_;
    // Sigma Points matrix
    thrust::device_vector<float> sigmaPoints_;
    // Weights for sigma points
    thrust::device_vector<float> weightsMean_;
    thrust::device_vector<float> weightsCovariance_;
    // Predicted State Mean
    thrust::device_vector<float> predictedStateMean_;
    // Predicted State Covarince 
    thrust::device_vector<float> predictedStateCovariance_;
    // Precited Measuement Sigma points
    thrust::device_vector<float> predictedMeasurementSigmaPoints_;
    // Precited Measurement mean
    thrust::device_vector<float> predictedMeasurementMean_;
    // Precited Measurement Covariance
    thrust::device_vector<float> predictedMeasurementCovariance_;
    // Cross-Covariance matrix between state and measurement
    thrust::device_vector<float> crossCovariance_;
    // Kalman Gain
    thrust::device_vector<float> kalmanGain_;
    // Functions
    void updateAllSigmaPoints(const std::vector<float>& measurement, const float dt); 
    void predictStateMean();
    void predictStateCovariance(); 
    thrust::device_vector<float> scaleMeasurementCovarianceMatrix();
    thrust::device_vector<float> choleskyDecompositionMeasurment(); 
    void generateMeasurmentSigmaPoints(const std::vector<float>& measurement, float dt);
    void predictMeasurementMean();
    void computeMeasurementCovariance();
    void computeCrossCovariance();
    thrust::device_vector<float> prepareMeasurementVector(const std::vector<float>& measurement, float dt);
    thrust::device_vector<float> computeInverseMeasurementCovariance();
    void computeKalmanGain();
    void updateStateWithMeasurement(const std::vector<float>& measurement, float dt);
    void updateStateCovariance();
public:
    UKF(const float & alpha, const float & beta, const float & kappa);
    ~UKF();
    // Init state for thrust::vector
    void initState(const std::vector<float>& state_vector);
    void initStateCovarianceMatrix(const float & positionVariance, const float & orientationVariance, const float & velocityVariance, const float & gyroBiasVariance, const float & accelBiasVariance);
    void initProcessCovarianceMatrix(const float & positionNoise, const float & orientationNoise, const float & velocityNoise, const float & gyroBiasNoise, const float & accelBiasNoise);
    void updateMeasurementCovarianceMatrix(const float positionVariance, const float orientationVariance, const float velocityVariance, const float angularVelocityVariance, const float linearAccelerationVariance);
    // sigma points generation functions
    void generateSigmaPoints();
    thrust::device_vector<float> scaleCovarianceMatrix() ;
    thrust::device_vector<float> choleskyDecomposition();
    void initializeWeights();
    void updatePrevPose(const std::vector<float>& newPos);
    void printStateInfo();
    void updateFilter(const std::vector<float>& measurement, const float dt, const float positionVariance, const float orientationVariance, const float velocityVariance, const float angularVelocityVariance, const float linearAccelerationVariance);
    std::vector<float> getState();
    std::vector<float> getStateCovarianceMatrix();
};

#endif