#include "cpuFilter.hpp"

UKF::UKF() : rosLogger_(rclcpp::get_logger("cuUKF"))
{
    // Initialize the state vector to a size of 16
    this->state_ = Eigen::VectorXd(16);
    this->state_.setZero(); // Initialize all elements to zero

    // Initialize UKF parameters
    this->alpha_ = 1e-3;
    this->beta_ = 2.0;
    this->kappa_ = 0.0;
    this->lambda_ = this->alpha_ * this->alpha_ * (this->n + this->kappa_) - this->n;
}

void UKF::initializeState(const Eigen::VectorXd& initialState, const Eigen::MatrixXd& pose_covariance, const Eigen::MatrixXd& ang_vel_covariance, const Eigen::MatrixXd& accel_covariance)
{
    // Check the dimension of initialState to ensure it matches your state vector size
    if (initialState.size() != state_.size()) throw std::runtime_error("Initial state size does not match the UKF state size.");

    // Set the initial state
    this->state_ = initialState;

    // Set the covaiance matrix
    this->initializeCovarianceMatrix_();
    this->updateNoiseMatrix(pose_covariance, ang_vel_covariance, accel_covariance);

    // Initalise sigma points and weights
    this->generateSigmaPoints_();
    this->initializeWeights_();

    // Log the initialization
    RCLCPP_INFO(this->rosLogger_, "UKF State initialized.");
}

void UKF::initializeCovarianceMatrix_()
{
    // Initialize the full 16x16 covariance matrix to zero
    this->P_ = Eigen::MatrixXd::Zero(16, 16);

    // Define standard variances
    double positionVariance = 0.1;    // Variance for position
    double orientationVariance = 0.01; // Variance for quaternion components
    double velocityVariance = 1.0;     // Variance for velocity
    double angularVelocityVariance = 0.01; // Variance for angular velocity
    double accelerationVariance = 0.1; // Variance for linear acceleration

    // Set covariance for position (3x3 block)
    this->P_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * positionVariance;

    // Set covariance for orientation (4x4 block)
    this->P_.block<4, 4>(3, 3) = Eigen::Matrix4d::Identity() * orientationVariance;

    // Set covariance for velocity (3x3 block)
    this->P_.block<3, 3>(7, 7) = Eigen::Matrix3d::Identity() * velocityVariance;

    // Set covariance for angular velocity (3x3 block)
    this->P_.block<3, 3>(10, 10) = Eigen::Matrix3d::Identity() * angularVelocityVariance;

    // Set covariance for linear acceleration (3x3 block)
    this->P_.block<3, 3>(13, 13) = Eigen::Matrix3d::Identity() * accelerationVariance;
}

void UKF::updateNoiseMatrix(const Eigen::MatrixXd& pose_covariance_6x6, const Eigen::MatrixXd& ang_vel_covariance_3x3,const Eigen::MatrixXd& accel_covariance_3x3)
{
    // Initialize the full 16x16 covariance matrix to zero
    this->R_ = Eigen::MatrixXd::Zero(16, 16);

    // Example variance values
    double orientationVariance = 0.01; // Orientation measurement noise variance
    double velocityVariance = 1.0; // Velocity measurement noise variance

    // Set variances for position (first 3 elements)
    this->R_.block<3, 3>(0, 0) = pose_covariance_6x6.block<3, 3>(0, 0);

    // Set variances for orientation (next 4 elements)
    this->R_.block<4, 4>(3, 3) = Eigen::Matrix4d::Identity() * orientationVariance;

    // Set variances for velocity (next 3 elements)
    this->R_.block<3, 3>(7, 7) = Eigen::Matrix3d::Identity() * velocityVariance;

    // Set variances for gyro bias (next 3 elements)
    this->R_.block<3, 3>(10, 10) = ang_vel_covariance_3x3;

    // Set variances for accelerometer bias (next 3 elements)
    this->R_.block<3, 3>(13, 13) = accel_covariance_3x3;
}

void UKF::generateSigmaPoints_()
{
    this->sigmaPoints_.resize(this->n, 2 * this->n + 1);

    // Compute scaling factor
    Eigen::MatrixXd S = (this->P_ * (this->n + this->lambda_)).llt().matrixL(); // Cholesky decomposition

    // Set the first column to the current state
    this->sigmaPoints_.col(0) = state_;

    // Set remaining sigma points
    for (int i = 0; i < n; ++i)
    {
        this->sigmaPoints_.col(i + 1)       = this->state_ + S.col(i);
        this->sigmaPoints_.col(i + 1 + n)   = this->state_ - S.col(i);
    }
}

void UKF::initializeWeights_()
{
    this->weightsMean_ = Eigen::VectorXd(2 * this->n + 1);
    this->weightsCovariance_ = Eigen::VectorXd(2 * this->n + 1);

    double weight_0 = this->lambda_ / (this->lambda_ + this->n);
    double weight_others = 1 / (2 * (this->lambda_ + this->n));

    this->weightsMean_(0) = weight_0;
    this->weightsCovariance_(0) = weight_0 + (1 - alpha_ * alpha_ + beta_);

    for (int i = 1; i < 2 * n + 1; ++i)
    {
        this->weightsMean_(i) = weight_others;
        this->weightsCovariance_(i) = weight_others;
    }
}

Eigen::Quaterniond UKF::updateOrientationSigmaPoint_(const Eigen::Quaterniond& current_orientation, const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& gyro_bias, double dt)
{
    Eigen::Vector3d correctedOmega = 0.5 * (angular_velocity - gyro_bias) * dt;
    double omega_norm = correctedOmega.norm();

    if (omega_norm <= std::numeric_limits<double>::epsilon()) return current_orientation;

    double half_theta = 0.5 * omega_norm;
    Eigen::Quaterniond delta_q_quaternion(Eigen::AngleAxisd(half_theta, correctedOmega.normalized()));

    // Perform the quaternion multiplication and normalize
    Eigen::Quaterniond updated_orientation = (delta_q_quaternion * current_orientation).normalized();

    return updated_orientation;
}

void UKF::updatePositionAndVelocitySigmaPoint_(Eigen::VectorXd& sigmaPoint, const Eigen::Vector3d& accel_meas, double dt)
{
    // Extract position, orientation, and velocity from the sigma point
    Eigen::Vector3d position = sigmaPoint.segment(0, 3);
    Eigen::Quaterniond orientation(sigmaPoint(3), sigmaPoint(4), sigmaPoint(5), sigmaPoint(6));
    Eigen::Vector3d velocity = sigmaPoint.segment(7, 3);
    Eigen::Vector3d accel_bias = sigmaPoint.segment(13, 3);

    // Compute the acceleration in the global frame
    Eigen::Vector3d accel_global = orientation * (accel_meas - accel_bias) - Eigen::Vector3d(0, 0, -9.81);  // Assuming gravity along z

    // Update velocity
    Eigen::Vector3d velocity_updated = velocity + accel_global * dt;

    // Update position
    Eigen::Vector3d position_updated = position + (velocity * dt) + (0.5 * accel_global * dt * dt);

    // Write the updated position and velocity back into the sigma point
    sigmaPoint.segment(0, 3) = position_updated;
    sigmaPoint.segment(7, 3) = velocity_updated;
}

void UKF::updateBiasesSigmaPoint_(Eigen::VectorXd& sigmaPoint, const Eigen::Vector3d& gyro_error, const Eigen::Vector3d& accel_error)
{
    // Extract gyro and accel biases from the sigma point
    Eigen::Vector3d gyro_bias = sigmaPoint.segment(10, 3);
    Eigen::Vector3d accel_bias = sigmaPoint.segment(13, 3);

    // Update biases based on the error estimates
    gyro_bias += gyro_error;
    accel_bias += accel_error;

    // Write the updated biases back into the sigma point
    sigmaPoint.segment(10, 3) = gyro_bias;
    sigmaPoint.segment(13, 3) = accel_bias;
}

void UKF::propagateSigmaPointsThroughProcessModel_(double dt, const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& accel_meas, const Eigen::Vector3d& gyro_error, const Eigen::Vector3d& accel_error)
{
    for (int i = 0; i < this->sigmaPoints_.cols(); ++i) 
    {
        // Create a copy of the i-th column of sigmaPoints_
        Eigen::VectorXd sigmaPoint = this->sigmaPoints_.col(i);

        // Extract current orientation from the sigma point
        Eigen::Quaterniond current_orientation(sigmaPoint(3), sigmaPoint(4), sigmaPoint(5), sigmaPoint(6));

        // Update orientation for this sigma point
        Eigen::Quaterniond updated_orientation = updateOrientationSigmaPoint_(current_orientation, angular_velocity, gyro_error, dt);
        updated_orientation.normalize();

        // Write back updated orientation into the sigma point
        sigmaPoint(3) = updated_orientation.w();
        sigmaPoint(4) = updated_orientation.x();
        sigmaPoint(5) = updated_orientation.y();
        sigmaPoint(6) = updated_orientation.z();

        // Update position and velocity for this sigma point
        this->updatePositionAndVelocitySigmaPoint_(sigmaPoint, accel_meas, dt);

        // Update biases for this sigma point using the error measurements
        this->updateBiasesSigmaPoint_(sigmaPoint, gyro_error, accel_error);

        // Write the updated sigma point back into the matrix
        this->sigmaPoints_.col(i) = sigmaPoint;
    }
}

void UKF::predictStateMean_()
{
    // Initialize the predicted state mean vector
    Eigen::VectorXd tempPredictedStateMean = Eigen::VectorXd::Zero(this->state_.size());

    // Compute the weighted sum of the sigma points
    for (int i = 0; i < this->sigmaPoints_.cols(); ++i) tempPredictedStateMean += this->weightsMean_(i) * this->sigmaPoints_.col(i);

    // Store the predicted state mean
    this->predictedStateMean_ = tempPredictedStateMean;
}

void UKF::predictStateCovariance_()
{
    // Initialize the predicted state covariance matrix
    Eigen::MatrixXd predictedStateCovariance = Eigen::MatrixXd::Zero(this->state_.size(), this->state_.size());

    // Compute the weighted sum of the outer products of the sigma point deviations
    for (int i = 0; i < this->sigmaPoints_.cols(); ++i)
    {
        Eigen::VectorXd deviation = this->sigmaPoints_.col(i) - this->state_; // Deviation of sigma point from predicted mean
        predictedStateCovariance += this->weightsCovariance_(i) * deviation * deviation.transpose();
    }

    // Update the state covariance matrix with the predicted state covariance
    this->P_ = predictedStateCovariance;
}

void UKF::updatePrevPose(const Eigen::Vector3d& prevPose)
{
    this->prevPosition_ = prevPose;
}

Eigen::VectorXd UKF::updateMeasurementModel(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation, const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& accel_meas, double dt, const Eigen::MatrixXd& pose_covariance_6x6, const Eigen::MatrixXd& ang_vel_covariance_3x3,const Eigen::MatrixXd& accel_covariance_3x3)
{
    // Update Noise Matrix R
    this->updateNoiseMatrix(pose_covariance_6x6, ang_vel_covariance_3x3, accel_covariance_3x3);

    Eigen::VectorXd measurement(16); // Adjust based on your measurement vector size

    // Position - directly from visual odometry
    measurement.head(3) = position;

    // Orientation - directly from visual odometry
    Eigen::Vector4d orientation_coeffs = Eigen::Vector4d(orientation.w(), orientation.x(), orientation.y(), orientation.z());
    measurement.segment(3, 4) = orientation_coeffs;

    // Derived Velocity
    Eigen::Vector3d derived_velocity = (position - this->prevPosition_) / dt;
    measurement.segment(7, 3) = derived_velocity;

    // Angular Velocity corrected for Gyro Bias
    Eigen::Vector3d gyro_bias = this->state_.segment(10, 3);
    measurement.segment(10, 3) = angular_velocity - gyro_bias;

    // Linear Acceleration corrected for Accelerometer Bias
    Eigen::Vector3d accel_bias = this->state_.segment(13, 3);
    measurement.segment(13, 3) = accel_meas - accel_bias;

    return measurement;
}

Eigen::VectorXd UKF::applyMeasurementModelToSigmaPoint_(const Eigen::VectorXd& sigmaPoint, const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& accel_meas)
{
    Eigen::VectorXd measurement(16); // Measurement vector size

    // Extract components from the sigma point
    Eigen::Vector3d position = sigmaPoint.segment(0, 3);
    Eigen::Quaterniond orientation(sigmaPoint(3), sigmaPoint(4), sigmaPoint(5), sigmaPoint(6));
    Eigen::Vector3d velocity = sigmaPoint.segment(7, 3); // Directly use velocity
    Eigen::Vector3d gyro_bias = sigmaPoint.segment(10, 3);
    Eigen::Vector3d accel_bias = sigmaPoint.segment(13, 3);

    // Position
    measurement.head(3) = position;

    // Orientation
    measurement.segment(3, 4) = Eigen::Vector4d(orientation.w(), orientation.x(), orientation.y(), orientation.z());

    // Velocity
    measurement.segment(7, 3) = velocity;

    // Angular Velocity corrected for Gyro Bias
    measurement.segment(10, 3) = angular_velocity - gyro_bias;

    // Linear Acceleration corrected for Accelerometer Bias
    measurement.segment(13, 3) = accel_meas - accel_bias;

    return measurement;
}

void UKF::transformSigmaPointsToMeasurementSpace_(const Eigen::Vector3d& angular_velocity, const Eigen::Vector3d& accel_meas)
{
    const int measurementDimension = 16; /* size of your measurement vector, e.g., 16 */
    this->measurementSigmaPoints_.resize(measurementDimension, this->sigmaPoints_.cols());

    for (int i = 0; i < sigmaPoints_.cols(); ++i) this->measurementSigmaPoints_.col(i) = this->applyMeasurementModelToSigmaPoint_(this->sigmaPoints_.col(i), angular_velocity, accel_meas);
}

void UKF::calculatePredictedMeasurementMean_()
{
    // Initialize the predicted measurement mean vector
    this->predictedMeasurementMean_ = Eigen::VectorXd::Zero(this->measurementSigmaPoints_.rows());

    // Compute the weighted sum of the measurement sigma points
    for (int i = 0; i < this->measurementSigmaPoints_.cols(); ++i) this->predictedMeasurementMean_ += this->weightsMean_(i) * this->measurementSigmaPoints_.col(i);
}

void UKF::calculatePredictedMeasurementCovariance_()
{
    int measurementDimension = this->predictedMeasurementMean_.size();
    this->predictedMeasurementCovariance_ = Eigen::MatrixXd::Zero(measurementDimension, measurementDimension);

    for (int i = 0; i < this->measurementSigmaPoints_.cols(); ++i) 
    {
        Eigen::VectorXd deviation = this->measurementSigmaPoints_.col(i) - this->predictedMeasurementMean_;
        this->predictedMeasurementCovariance_ += this->weightsCovariance_(i) * (deviation * deviation.transpose());
    }

    // Add the measurement noise covariance
    this->predictedMeasurementCovariance_ += this->R_;
}

void UKF::calculateCrossCovariance_()
{
    int stateDimension = this->predictedStateMean_.size();
    int measurementDimension = this->predictedMeasurementMean_.size();
    this->Pxy_ = Eigen::MatrixXd::Zero(stateDimension, measurementDimension);

    for (int i = 0; i < this->sigmaPoints_.cols(); ++i) 
    {
        Eigen::VectorXd stateDeviation = this->sigmaPoints_.col(i) - this->predictedStateMean_;
        Eigen::VectorXd measurementDeviation = this->measurementSigmaPoints_.col(i) - this->predictedMeasurementMean_;
        this->Pxy_ += weightsCovariance_(i) * (stateDeviation * measurementDeviation.transpose());
    }
}

void UKF::computeKalmanGain_()
{
    this->K_ = this->Pxy_ * this->predictedMeasurementCovariance_.inverse();
}

void UKF::updateStateAndCovariance_(const Eigen::VectorXd& actualMeasurement)
{
    // Calculate the measurement residual
    Eigen::VectorXd measurementResidual = actualMeasurement - this->predictedMeasurementMean_;

    // Update state estimate using the Kalman Gain
    this->state_ = this->predictedStateMean_ + this->K_ * measurementResidual;

    // Extract quaternion components (w, x, y, z) from the state
    Eigen::Quaterniond updated_orientation(this->state_(3), this->state_(4), this->state_(5), this->state_(6));

    // Normalize the quaternion
    updated_orientation.normalize();

    // Put the normalized quaternion back into the state vector
    this->state_(3) = updated_orientation.w();
    this->state_(4) = updated_orientation.x();
    this->state_(5) = updated_orientation.y();
    this->state_(6) = updated_orientation.z();

    std::cout << this->state_ << "\n" << std::endl;

    // Update state covariance matrix
    this->P_ = this->P_ - this->K_ * this->predictedMeasurementCovariance_ * this->K_.transpose();
}

UKF::~UKF()
{
}