#include "gpuFilter.cuh"


// --------------- GPU Kernels ---------------------- //

__global__ void scaleCovarianceMatrixKernel(float* P, int n, float alpha, float kappa) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n*n) P[idx] *= (alpha * (n + kappa));
}

__global__ void generateSigmaPointsKernel(float* sigmaPoints, const float* state, const float* cholP, int n, float lambda_plus_n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 * n + 1) return;

    int state_idx = idx * n; // Index for the beginning of the current sigma point in the flattened array

    // For the 0th sigma point, copy the state directly
    if (idx == 0) 
    {
        for (int i = 0; i < n; ++i) sigmaPoints[state_idx + i] = state[i];
    } 
    else 
    {
        int chol_idx = (idx - 1) % n; // Index for the current column of the Cholesky matrix
        float sign = idx <= n ? 1 : -1; // Determine whether to add or subtract

        for (int i = 0; i < n; ++i) sigmaPoints[state_idx + i] = state[i] + sign * sqrt(lambda_plus_n) * cholP[i * n + chol_idx];
    }

    // Normalize the quaternion if this sigma point contains one
    if (n > 6) // Assuming quaternion starts at index 3 and ends at index 6
    {
        float norm = sqrt(sigmaPoints[state_idx + 3] * sigmaPoints[state_idx + 3] + sigmaPoints[state_idx + 4] * sigmaPoints[state_idx + 4] + sigmaPoints[state_idx + 5] * sigmaPoints[state_idx + 5] + sigmaPoints[state_idx + 6] * sigmaPoints[state_idx + 6]);
        for (int i = 3; i <= 6; ++i) sigmaPoints[state_idx + i] /= norm;
    }
}

__global__ void initializeWeightsKernel(float* weightsMean, float* weightsCovariance, int n, float lambda, float alpha, float beta)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= 2 * n + 1) return;

    if (idx == 0)
    {
        weightsMean[idx] = lambda / (n + lambda);
        weightsCovariance[idx] = weightsMean[idx] + (1 - alpha * alpha + beta);
    }
    else
    {
        weightsMean[idx] = 1.0 / (2 * (n + lambda));
        weightsCovariance[idx] = 1.0 / (2 * (n + lambda));
    }
}

__global__ void updateSigmaPointsKernel(float* sigmaPoints, const float* measurement, int sigmaPointCount, float dt, int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sigmaPointCount) return;

    int baseIdx = idx * n;

    // Angular velocity components updated in the previous step
    float wx = sigmaPoints[baseIdx + 10];
    float wy = sigmaPoints[baseIdx + 11];
    float wz = sigmaPoints[baseIdx + 12];

    // Current quaternion components
    float qw = sigmaPoints[baseIdx + 3];
    float qx = sigmaPoints[baseIdx + 4];
    float qy = sigmaPoints[baseIdx + 5];
    float qz = sigmaPoints[baseIdx + 6];

    // Quaternion derivative due to angular velocity
    float dot_qw = 0.5 * (-qx * wx - qy * wy - qz * wz);
    float dot_qx = 0.5 * (qw * wx + qy * wz - qz * wy);
    float dot_qy = 0.5 * (qw * wy - qx * wz + qz * wx);
    float dot_qz = 0.5 * (qw * wz + qx * wy - qy * wx);

    // Update quaternion components with Euler integration
    sigmaPoints[baseIdx + 3] += dot_qw * dt;
    sigmaPoints[baseIdx + 4] += dot_qx * dt;
    sigmaPoints[baseIdx + 5] += dot_qy * dt;
    sigmaPoints[baseIdx + 6] += dot_qz * dt;

    // Normalize the updated quaternion
    float norm = sqrt(sigmaPoints[baseIdx + 3] * sigmaPoints[baseIdx + 3] +
                      sigmaPoints[baseIdx + 4] * sigmaPoints[baseIdx + 4] +
                      sigmaPoints[baseIdx + 5] * sigmaPoints[baseIdx + 5] +
                      sigmaPoints[baseIdx + 6] * sigmaPoints[baseIdx + 6]);

    sigmaPoints[baseIdx + 3] /= norm;
    sigmaPoints[baseIdx + 4] /= norm;
    sigmaPoints[baseIdx + 5] /= norm;
    sigmaPoints[baseIdx + 6] /= norm;

    // Directly update the sigma point's angular velocity by adding the difference between measured and current angular velocity
    sigmaPoints[baseIdx + 10] += measurement[10] - sigmaPoints[baseIdx + 10];
    sigmaPoints[baseIdx + 11] += measurement[11] - sigmaPoints[baseIdx + 11];
    sigmaPoints[baseIdx + 12] += measurement[12] - sigmaPoints[baseIdx + 12];

    // Extract current position, velocity, and acceleration
    float x = sigmaPoints[baseIdx + 0];
    float y = sigmaPoints[baseIdx + 1];
    float z = sigmaPoints[baseIdx + 2];
    float vx = sigmaPoints[baseIdx + 7];
    float vy = sigmaPoints[baseIdx + 8];
    float vz = sigmaPoints[baseIdx + 9];
    float ax = sigmaPoints[baseIdx + 13];
    float ay = sigmaPoints[baseIdx + 14];
    float az = sigmaPoints[baseIdx + 15] - 9.81f; // Adjust for gravity

    // Update position using the equation of motion
    sigmaPoints[baseIdx + 0] = x + vx * dt + 0.5f * ax * dt * dt; // Update x position
    sigmaPoints[baseIdx + 1] = y + vy * dt + 0.5f * ay * dt * dt; // Update y position
    sigmaPoints[baseIdx + 2] = z + vz * dt + 0.5f * az * dt * dt; // Update z position

    // Update velocity using the equation of motion v = u + at
    sigmaPoints[baseIdx + 7] = vx + ax * dt; // Update vx
    sigmaPoints[baseIdx + 8] = vy + ay * dt; // Update vy
    sigmaPoints[baseIdx + 9] = vz + az * dt; // Update vz, considering gravity if az includes it

    // Directly update the sigma point's linear acceleration by adding the difference between measured and current linear acceleration
    sigmaPoints[baseIdx + 13] += measurement[13] - sigmaPoints[baseIdx + 13]; // ax
    sigmaPoints[baseIdx + 14] += measurement[14] - sigmaPoints[baseIdx + 14]; // ay
    sigmaPoints[baseIdx + 15] += measurement[15] - sigmaPoints[baseIdx + 15]; // az
}

__global__ void predictStateMeanKernel(const float* sigmaPoints, const float* weightsMean, float* predictedStateMean, int stateDimension, int sigmaPointCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < stateDimension)
    {
        float sum = 0.0f;
        for (int j = 0; j < sigmaPointCount; ++j) sum += sigmaPoints[j * stateDimension + idx] * weightsMean[j];
        predictedStateMean[idx] = sum;
    }
}

__global__ void predictStateCovarianceKernel(float* predictedSigmaPoints, float* predictedStateMean, float* weightsCovariance, float* processNoiseCovariance, float* predictedStateCovariance, int n, int sigmaPointCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int jdx = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < n && jdx < n)
    {
        float covSum = 0.0f;
        for (int i = 0; i < sigmaPointCount; ++i)
        {
            float diff_i_j = (predictedSigmaPoints[i * n + idx] - predictedStateMean[idx]) * (predictedSigmaPoints[i * n + jdx] - predictedStateMean[jdx]);
            covSum += weightsCovariance[i] * diff_i_j;
        }
        if (idx == jdx) predictedStateCovariance[idx * n + jdx] = covSum + processNoiseCovariance[idx * n + jdx];
        else predictedStateCovariance[idx * n + jdx] = covSum;
    }
}

__global__ void predictMeasurementMeanKernel(const float* sigmaPoints, const float* weightsMean, float* predictedMeasurementMean, int measurementDimension, int sigmaPointCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < measurementDimension) 
    {
        float mean = 0.0f;
        for (int i = 0; i < sigmaPointCount; ++i) mean += sigmaPoints[i * measurementDimension + idx] * weightsMean[i];
        predictedMeasurementMean[idx] = mean;
    }
}

__global__ void computeMeasurementCovarianceKernel(const float* predictedMeasurementSigmaPoints, const float* predictedMeasurementMean, float* predictedMeasurementCovariance, const float* measurementNoiseCovariance, int n, int sigmaPointCount, const float* weightsCovariance) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int jdx = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < n && jdx < n) 
    {
        float covariance = 0.0f;
        for (int k = 0; k < sigmaPointCount; ++k)
        {
            float diff_i = predictedMeasurementSigmaPoints[k * n + idx] - predictedMeasurementMean[idx];
            float diff_j = predictedMeasurementSigmaPoints[k * n + jdx] - predictedMeasurementMean[jdx];
            covariance += weightsCovariance[k] * diff_i * diff_j;
        }

        if (idx == jdx) covariance += measurementNoiseCovariance[idx * n + jdx];
        predictedMeasurementCovariance[idx * n + jdx] = covariance;
    }
}

__global__ void computeCrossCovarianceKernel(const float* predictedStateSigmaPoints, const float* predictedMeasurementSigmaPoints, const float* predictedStateMean, const float* predictedMeasurementMean, float* crossCovariance, const float* weightsCovariance, int stateDimension, int measurementDimension, int sigmaPointCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int jdx = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < stateDimension && jdx < measurementDimension)
    {
        float covSum = 0.0f;
        for (int i = 0; i < sigmaPointCount; ++i)
        {
            float diffState = predictedStateSigmaPoints[i * stateDimension + idx] - predictedStateMean[idx];
            float diffMeasurement = predictedMeasurementSigmaPoints[i * measurementDimension + jdx] - predictedMeasurementMean[jdx];
            covSum += weightsCovariance[i] * diffState * diffMeasurement;
        }
        crossCovariance[idx * measurementDimension + jdx] = covSum;
    }
}

__global__ void computeKalmanGainKernel(const float* crossCovariance, const float* predictedMeasurementCovarianceInv, float* kalmanGain, int stateDim, int measDim)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int jdx = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < stateDim && jdx < measDim) 
    {
        float sum = 0.0;
        for (int k = 0; k < measDim; ++k) sum += crossCovariance[idx * measDim + k] * predictedMeasurementCovarianceInv[k * measDim + jdx];
        kalmanGain[idx * measDim + jdx] = sum;
    }
}

__global__ void updateStateWithMeasurementKernel(float* stateMean, const float* kalmanGain, const float* measurementResidual, int stateDim, int measDim) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < stateDim) 
    {
        float updateValue = 0.0;
        for (int k = 0; k < measDim; ++k) updateValue += kalmanGain[idx * measDim + k] * measurementResidual[k];
        stateMean[idx] += updateValue; // Update the state with the computed value
    }
}

__global__ void normalizeQuaternionKernel(float* stateData) 
{
    // Since this operation is lightweight, we'll use a single thread.
    if (threadIdx.x == 0) 
    {
        int quaternionStartIdx = 3; // Starting index of quaternion components in the state vector
        float norm = sqrtf(stateData[quaternionStartIdx] * stateData[quaternionStartIdx] + stateData[quaternionStartIdx + 1] * stateData[quaternionStartIdx + 1] + stateData[quaternionStartIdx + 2] * stateData[quaternionStartIdx + 2] + stateData[quaternionStartIdx + 3] * stateData[quaternionStartIdx + 3]);
        
        // Normalize each component of the quaternion
        for (int i = 0; i < 4; ++i) stateData[quaternionStartIdx + i] /= norm;
    }
}

// --------------- GPU Kernels ---------------------- //


UKF::UKF(const float & alpha, const float & beta, const float & kappa) : rosLogger_(rclcpp::get_logger("cuUKF"))
{
    // Initialize UKF parameters
    this->alpha_ = alpha;
    this->beta_ = beta;
    this->kappa_ = kappa;
    this->lambda_ = this->alpha_ * this->alpha_ * (this->n + this->kappa_) - this->n;

    // Initialize prevPose_ to store 3 floats
    this->prevPose_.resize(3, 0.0f);

    // cuBLAS handle initialization
    cublasStatus_t cublasStatus = cublasCreate(&this->cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) RCLCPP_ERROR(this->rosLogger_, "cuBLAS handle creation failed");

    // cuSOLVER handle initialization
    cusolverStatus_t cusolverStatus = cusolverDnCreate(&this->cusolverHandle);
    if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) RCLCPP_ERROR(this->rosLogger_, "cuSOLVER handle creation failed");

    // Initialize State
    this->state_ = thrust::device_vector<float>(this->n);

    // Initialize State Covaraince matrix
    this->P_ = thrust::device_vector<float>(this->n * this->n);

    // Initialize Process Covairance matrix
    this->Q_ = thrust::device_vector<float>(this->n * this->n);

    // Initalize Measuremnt Covariance matrix
    this->R_ = thrust::device_vector<float>(this->n * this->n);

    RCLCPP_INFO(this->rosLogger_, "GPU UKF Initalized!!!");
}

UKF::~UKF()
{
    cublasDestroy(this->cublasHandle);
    cusolverDnDestroy(this->cusolverHandle);
    RCLCPP_INFO(this->rosLogger_, "CUDA handles destroyed");
}

void UKF::initState(const std::vector<float>& state_vector)
{
    thrust::copy(state_vector.begin(), state_vector.end(), this->state_.begin());
}

void UKF::initStateCovarianceMatrix(const float & positionVariance, const float & orientationVariance, const float & linearVelocityVariance, const float & angularVelocityVariance, const float & linearAccelerationVariance)
{
    // Initialize all elements to zero
    thrust::fill(this->P_.begin(), this->P_.end(), 0.0f);

    // Set variances for each part of the state
    // Position
    for (int i = 0; i < 3; ++i) this->P_[i * this->n + i] = positionVariance;

    // Orientation
    for (int i = 3; i < 7; ++i) this->P_[i * this->n + i] = orientationVariance;

    // Linear Velocity
    for (int i = 7; i < 10; ++i) this->P_[i * this->n + i] = linearVelocityVariance;

    // Angular Velocity
    for (int i = 10; i < 13; ++i) this->P_[i * this->n + i] = angularVelocityVariance;

    // Linear Acceleration
    for (int i = 13; i < 16; ++i) this->P_[i * this->n + i] = linearAccelerationVariance;
}

void UKF::initProcessCovarianceMatrix(const float & positionNoise, const float & orientationNoise, const float & linearVelocityNoise, const float & angularVelocityNoise, const float & linearAccelerationNoise)
{
    // Assuming Q_ is a 16x16 matrix for a state vector with 16 elements
    thrust::fill(this->Q_.begin(), this->Q_.end(), 0.0f);

    // Position-related noise
    for (int i = 0; i < 3; ++i) this->Q_[i * this->n + i] = positionNoise;

    // Orientation-related noise (quaternion)
    for (int i = 3; i < 7; ++i) this->Q_[i * this->n + i] = orientationNoise;

    // Linear Velocity Noise
    for (int i = 7; i < 10; ++i) this->Q_[i * this->n + i] = linearVelocityNoise;

    // Angular Velocity Noise
    for (int i = 10; i < 13; ++i) this->Q_[i * this->n + i] = angularVelocityNoise;

    // Linear Acceleration Noise
    for (int i = 13; i < 16; ++i) this->Q_[i * this->n + i] = linearAccelerationNoise;
}

void UKF::updateMeasurementCovarianceMatrix(const float positionVariance, const float orientationVariance, const float velocityVariance, const float angularVelocityVariance, const float linearAccelerationVariance)
{
    // Reset R_ to zeros, assuming R_ is for a 16x16 matrix
    thrust::fill(this->R_.begin(), this->R_.end(), 0.0f);

    // Update position covariance
    for (int i = 0; i < 3; ++i) this->R_[i * this->n + i] = positionVariance;
    
    // Update orientation covariance for quaternion components (w, x, y, z)
    for (int i = 3; i < 7; ++i) this->R_[i * this->n + i] = orientationVariance;

    // Update linear velocity covariance
    for (int i = 7; i < 10; ++i) this->R_[i * this->n + i] = velocityVariance;

    // Update angular velocity covariance
    for (int i = 10; i < 13; ++i) this->R_[i * this->n + i] = angularVelocityVariance;

    // Update linear acceleration covariance
    for (int i = 13; i < 16; ++i) this->R_[i * this->n + i] = linearAccelerationVariance;
}

thrust::device_vector<float> UKF::scaleCovarianceMatrix() 
{
    // Compute the scale factor using class variables alpha_ and kappa_
    float scale = this->alpha_ * (this->n + this->kappa_);

    // Launch the CUDA kernel to scale the covariance matrix P_
    int numThreadsPerBlock = 256; // You can adjust this based on your GPU's capabilities
    int numBlocks = (this->n * this->n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    thrust::device_vector<float> P_copy = this->P_;

    // Call the CUDA kernel
    scaleCovarianceMatrixKernel<<<numBlocks, numThreadsPerBlock>>>(thrust::raw_pointer_cast(P_copy.data()), this->n, this->alpha_, this->kappa_);

    // Synchronize to ensure the kernel has finished execution
    cudaDeviceSynchronize();

    return P_copy;
}

thrust::device_vector<float> UKF::choleskyDecomposition()
{
    // Check if the cuSOLVER handle is valid
    if (this->cusolverHandle == nullptr) RCLCPP_ERROR(this->rosLogger_, "cuSOLVER handle is not valid. Make sure it was initialized correctly.");

    // Create a copy of the state covariance matrix P_ to perform the Cholesky decomposition
    thrust::device_vector<float> P_scaled = this->scaleCovarianceMatrix();

    // Declare integer pointer for devInfo
    int* devInfo = nullptr;

    // Allocate memory for devInfo
    cudaMalloc((void**)&devInfo, sizeof(int));

    // Determine workspace size for Cholesky decomposition
    int lwork = 0;
    cusolverDnSpotrf_bufferSize(this->cusolverHandle, CUBLAS_FILL_MODE_LOWER, this->n, thrust::raw_pointer_cast(P_scaled.data()), this->n, &lwork);

    // Allocate workspace
    float* d_work = nullptr;
    cudaMalloc((void**)&d_work, sizeof(float) * lwork);

    // Perform Cholesky decomposition
    cusolverStatus_t status = cusolverDnSpotrf(this->cusolverHandle, CUBLAS_FILL_MODE_LOWER, this->n, thrust::raw_pointer_cast(P_scaled.data()), this->n, d_work, lwork, devInfo);

    // Check the value of devInfo for any errors
    int devInfo_h = 0;
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != CUSOLVER_STATUS_SUCCESS || devInfo_h != 0) RCLCPP_ERROR(this->rosLogger_, "Cholesky decomposition failed.");

    // Free the workspace memory and devInfo
    cudaFree(d_work);
    cudaFree(devInfo);

    cudaDeviceSynchronize();
    
    return P_scaled;
}

void UKF::generateSigmaPoints()
{
    // Ensure sigmaPoints_ is correctly sized
    this->sigmaPoints_.resize((2 * this->n + 1) * this->n);

    // Calculate lambda_plus_n
    float lambda_plus_n = this->lambda_ + this->n;

    // Set up kernel execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = ((2 * this->n + 1) + threadsPerBlock - 1) / threadsPerBlock;

    // Get scaled and decomposed State covariance matrix from Cholesky decomposition
    thrust::device_vector<float> P_scaled = this->choleskyDecomposition();

    // Launch kernel to generate sigma points, now passing lambda_plus_n as an additional argument
    generateSigmaPointsKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(this->sigmaPoints_.data()), thrust::raw_pointer_cast(this->state_.data()), thrust::raw_pointer_cast(P_scaled.data()), this->n, lambda_plus_n);

    // Wait for kernel to finish
    cudaDeviceSynchronize();
}

void UKF::initializeWeights()
{
    // Ensure the device vectors have the correct size
    this->weightsMean_.resize(2 * this->n + 1);
    this->weightsCovariance_.resize(2 * this->n + 1);

    // Calculate grid and block sizes for the kernel launch
    int threadsPerBlock = 256; // You may adjust this based on your GPU's capabilities
    int blocksPerGrid = ((2 * this->n + 1) + threadsPerBlock - 1) / threadsPerBlock; // Ensure enough blocks to cover all elements

    // Launch the kernel to initialize the weights on the device
    initializeWeightsKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(this->weightsMean_.data()), thrust::raw_pointer_cast(this->weightsCovariance_.data()), this->n, this->lambda_, this->alpha_, this->beta_);

    // Synchronize to ensure the kernel has finished execution
    cudaDeviceSynchronize();
}

void UKF::updatePrevPose(const std::vector<float>& newPos)
{
    thrust::copy(newPos.begin(), newPos.begin() + 3, this->prevPose_.begin());
}

void UKF::updateAllSigmaPoints(const std::vector<float>& measurement, const float dt) 
{
    thrust::device_vector<float> measurement_vector = this->prepareMeasurementVector(measurement, dt);

    int threadsPerBlock = 256;
    int blocksPerGrid = ((this->sigmaPointCount) + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the unified update kernel
    updateSigmaPointsKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(this->sigmaPoints_.data()), thrust::raw_pointer_cast(measurement_vector.data()), this->sigmaPointCount, dt, this->n);
    cudaDeviceSynchronize(); // Wait for the kernel to complete
}

void UKF::predictStateMean()
{
    // Ensure the predicted state mean vector is correctly sized
    this->predictedStateMean_.resize(this->n); // Assuming 'n' is the state dimension

    int threadsPerBlock = 256;
    int blocksPerGrid = (this->n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    predictStateMeanKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(this->sigmaPoints_.data()), thrust::raw_pointer_cast(this->weightsMean_.data()), thrust::raw_pointer_cast(this->predictedStateMean_.data()), this->n, this->sigmaPointCount);

    // Synchronize to ensure the kernel execution completes
    cudaDeviceSynchronize();
} 

void UKF::predictStateCovariance()
{
    this->predictedStateCovariance_.resize(this->n * this->n);
    int threadsPerBlockDim = 16;  // Using a 2D block for matrix operations
    dim3 threadsPerBlock(threadsPerBlockDim, threadsPerBlockDim);
    int blocksPerGridDim = (this->n + threadsPerBlockDim - 1) / threadsPerBlockDim;
    dim3 blocksPerGrid(blocksPerGridDim, blocksPerGridDim);

    // Launching the kernel
    predictStateCovarianceKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(this->sigmaPoints_.data()), thrust::raw_pointer_cast(this->predictedStateMean_.data()), thrust::raw_pointer_cast(this->weightsCovariance_.data()), thrust::raw_pointer_cast(this->Q_.data()), thrust::raw_pointer_cast(this->predictedStateCovariance_.data()), this->n, this->sigmaPointCount);
    cudaDeviceSynchronize();  // Ensure completion
    cudaMemcpy(thrust::raw_pointer_cast(this->P_.data()), thrust::raw_pointer_cast(this->predictedStateCovariance_.data()), this->n * this->n * sizeof(float), cudaMemcpyDeviceToDevice);
} 

thrust::device_vector<float> UKF::scaleMeasurementCovarianceMatrix() 
{
    // Compute the scale factor using class variables alpha_ and kappa_
    float scale = this->alpha_ * (this->n + this->kappa_);

    // Launch the CUDA kernel to scale the covariance matrix P_
    int numThreadsPerBlock = 256; // You can adjust this based on your GPU's capabilities
    int numBlocks = (this->n * this->n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    thrust::device_vector<float> R_copy = this->R_;

    // Call the CUDA kernel
    scaleCovarianceMatrixKernel<<<numBlocks, numThreadsPerBlock>>>(thrust::raw_pointer_cast(R_copy.data()), this->n, this->alpha_, this->kappa_);

    // Synchronize to ensure the kernel has finished execution
    cudaDeviceSynchronize();

    return R_copy;
}

thrust::device_vector<float> UKF::choleskyDecompositionMeasurment()
{
    // Check if the cuSOLVER handle is valid
    if (this->cusolverHandle == nullptr) RCLCPP_ERROR(this->rosLogger_, "cuSOLVER handle is not valid. Make sure it was initialized correctly.");

    // Create a copy of the state covariance matrix P_ to perform the Cholesky decomposition
    thrust::device_vector<float> R_scaled = this->scaleMeasurementCovarianceMatrix();

    // Declare integer pointer for devInfo
    int* devInfo = nullptr;

    // Allocate memory for devInfo
    cudaMalloc((void**)&devInfo, sizeof(int));

    // Determine workspace size for Cholesky decomposition
    int lwork = 0;
    cusolverDnSpotrf_bufferSize(this->cusolverHandle, CUBLAS_FILL_MODE_LOWER, this->n, thrust::raw_pointer_cast(R_scaled.data()), this->n, &lwork);

    // Allocate workspace
    float* d_work = nullptr;
    cudaMalloc((void**)&d_work, sizeof(float) * lwork);

    // Perform Cholesky decomposition
    cusolverStatus_t status = cusolverDnSpotrf(this->cusolverHandle, CUBLAS_FILL_MODE_LOWER, this->n, thrust::raw_pointer_cast(R_scaled.data()), this->n, d_work, lwork, devInfo);

    // Check the value of devInfo for any errors
    int devInfo_h = 0;
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != CUSOLVER_STATUS_SUCCESS || devInfo_h != 0) RCLCPP_ERROR(this->rosLogger_, "Cholesky decomposition failed.");

    // Free the workspace memory and devInfo
    cudaFree(d_work);
    cudaFree(devInfo);

    cudaDeviceSynchronize();
    
    return R_scaled;
}

void UKF::generateMeasurmentSigmaPoints(const std::vector<float>& measurement, float dt)
{
    // Measurement
    thrust::device_vector<float> newMeasurement= this->prepareMeasurementVector(measurement, dt);

    // Ensure sigmaPoints_ is correctly sized
    this->predictedMeasurementSigmaPoints_.resize(this->sigmaPointCount * this->n);

    // Calculate lambda_plus_n
    float lambda_plus_n = this->lambda_ + this->n;

    // Set up kernel execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = ((this->sigmaPointCount) + threadsPerBlock - 1) / threadsPerBlock;

    // Get scaled and decomposed State covariance matrix from Cholesky decomposition
    thrust::device_vector<float> R_scaled = this->choleskyDecompositionMeasurment();

    // Launch kernel to generate sigma points, now passing lambda_plus_n as an additional argument
    generateSigmaPointsKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(this->predictedMeasurementSigmaPoints_.data()), thrust::raw_pointer_cast(newMeasurement.data()), thrust::raw_pointer_cast(R_scaled.data()), this->n, lambda_plus_n);

    // Wait for kernel to finish
    cudaDeviceSynchronize();
}

void UKF::predictMeasurementMean() 
{
    // Assuming 'n' represents the dimension of your measurement vector
    this->predictedMeasurementMean_.resize(this->n, 0.0f);  // Resize each time to ensure correct size

    // Setup kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (this->n + threadsPerBlock - 1) / threadsPerBlock;

    // Call the kernel
    predictMeasurementMeanKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(this->predictedMeasurementSigmaPoints_.data()), thrust::raw_pointer_cast(this->weightsMean_.data()), thrust::raw_pointer_cast(this->predictedMeasurementMean_.data()), this->n, this->sigmaPointCount);

    cudaDeviceSynchronize();  // Ensure completion
}


void UKF::computeMeasurementCovariance() 
{
    // Resize and initialize the predictedMeasurementCovariance_ vector
    this->predictedMeasurementCovariance_.resize(this->n * this->n, 0.0f);

    dim3 threadsPerBlock(16, 16); // 2D block for matrix operations
    dim3 blocksPerGrid((this->n + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    computeMeasurementCovarianceKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(this->predictedMeasurementSigmaPoints_.data()), thrust::raw_pointer_cast(this->predictedMeasurementMean_.data()), thrust::raw_pointer_cast(this->predictedMeasurementCovariance_.data()), thrust::raw_pointer_cast(this->R_.data()), this->n, this->sigmaPointCount, thrust::raw_pointer_cast(this->weightsCovariance_.data()));

    cudaDeviceSynchronize(); // Ensure the kernel execution completes
}

void UKF::computeCrossCovariance()
{
    // Make sure the crossCovariance_ is resized appropriately
    this->crossCovariance_.resize(this->n * this->n, 0.0f); // Ensures it's correctly sized every time the function is called

    dim3 threadsPerBlock(16, 16); // Or other suitable configuration
    dim3 blocksPerGrid((this->n + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    computeCrossCovarianceKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(this->sigmaPoints_.data()), thrust::raw_pointer_cast(this->predictedMeasurementSigmaPoints_.data()), thrust::raw_pointer_cast(this->predictedStateMean_.data()), thrust::raw_pointer_cast(this->predictedMeasurementMean_.data()), thrust::raw_pointer_cast(this->crossCovariance_.data()), thrust::raw_pointer_cast(this->weightsCovariance_.data()), this->n, this->n, this->sigmaPointCount);

    cudaDeviceSynchronize(); // Wait for CUDA to finish
}

thrust::device_vector<float> UKF::prepareMeasurementVector(const std::vector<float>& measurement, float dt)
{
    std::vector<float> updatedMeasurement = measurement;

    // Compute derived velocity for each position dimension and update the measurement vector
    // Assuming the positions are stored at indices 0, 1, 2, and derived velocities should be placed at indices 7, 8, 9 for example
    for (size_t i = 0; i < 3; ++i)
    {
        float derivedVelocity = (measurement[i] - prevPose_[i]) / dt;
        updatedMeasurement[i + 7] = derivedVelocity; // Update with derived velocity
    }

    // Convert updatedMeasurement to device_vector and return
    return thrust::device_vector<float>(updatedMeasurement.begin(), updatedMeasurement.end());
}


thrust::device_vector<float> UKF::computeInverseMeasurementCovariance()
{
    thrust::device_vector<float> predictedMeasurementCovarianceInv(this->n * this->n); // Allocate space for the inverse matrix

    // Make a copy of the predictedMeasurementCovariance_ to preserve the original
    thrust::device_vector<float> predictedMeasurementCovarianceCopy = this->predictedMeasurementCovariance_;

    int *d_ipiv; // Pivot indices for LU decomposition
    int *d_info; // Info about the execution
    float *d_work; // Workspace for getrf
    int lwork = 0; // Size of workspace
    cusolverDnHandle_t cusolverHandle = this->cusolverHandle;

    cudaMalloc(&d_ipiv, this->n * sizeof(int));
    cudaMalloc(&d_info, sizeof(int));

    // Step 1: Query working space of getrf
    cusolverDnSgetrf_bufferSize(cusolverHandle, this->n, this->n, thrust::raw_pointer_cast(predictedMeasurementCovarianceCopy.data()), this->n, &lwork);
    cudaMalloc(&d_work, lwork * sizeof(float));

    // Step 2: LU decomposition on the copy
    cusolverDnSgetrf(cusolverHandle, this->n, this->n, thrust::raw_pointer_cast(predictedMeasurementCovarianceCopy.data()), this->n, d_work, d_ipiv, d_info);

    // Prepare identity matrix as the right-hand side
    thrust::device_vector<float> identity(this->n * this->n, 0);
    for (int i = 0; i < this->n; ++i) identity[i * this->n + i] = 1.0f;

    // Step 3: Solve linear equations for each column of the identity matrix to compute the inverse
    for (int i = 0; i < this->n; ++i) 
    {
        float* column = thrust::raw_pointer_cast(&identity[i * this->n]);
        cusolverDnSgetrs(cusolverHandle, CUBLAS_OP_N, this->n, 1, thrust::raw_pointer_cast(predictedMeasurementCovarianceCopy.data()), this->n, d_ipiv, column, this->n, d_info);
    }

    // Copy the inverse from identity matrix to predictedMeasurementCovarianceInv
    thrust::copy(identity.begin(), identity.end(), predictedMeasurementCovarianceInv.begin());

    // Cleanup
    cudaFree(d_ipiv);
    cudaFree(d_info);
    cudaFree(d_work);

    return predictedMeasurementCovarianceInv;
}

void UKF::computeKalmanGain() 
{
    // Ensure kalmanGain_ is correctly sized
    this->kalmanGain_.resize(this->n * this->n, 0.0f);

    // Compute the inverse of the predicted measurement covariance matrix
    thrust::device_vector<float> predictedMeasurementCovarianceInv = this->computeInverseMeasurementCovariance();

    // Setup kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((this->n + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    computeKalmanGainKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(this->crossCovariance_.data()), thrust::raw_pointer_cast(predictedMeasurementCovarianceInv.data()), thrust::raw_pointer_cast(this->kalmanGain_.data()), this->n, this->n);
    cudaDeviceSynchronize(); // Wait for CUDA to finish
}

void UKF::updateStateWithMeasurement(const std::vector<float>& measurement, float dt) 
{
    thrust::device_vector<float> actualMeasurement = this->prepareMeasurementVector(measurement, dt);
    thrust::device_vector<float> measurementResidual(this->n); // Ensure measDim is defined correctly
    thrust::transform(actualMeasurement.begin(), actualMeasurement.end(), this->predictedMeasurementMean_.begin(), measurementResidual.begin(), thrust::minus<float>());

    // Update the state with the measurement residual
    // blockSize and numBlocks need to be defined appropriately
    const int blockSize = 256; // A common choice for many CUDA-capable GPUs
    const int numBlocks = (this->n + blockSize - 1) / blockSize;
    updateStateWithMeasurementKernel<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(this->state_.data()), thrust::raw_pointer_cast(this->kalmanGain_.data()), thrust::raw_pointer_cast(measurementResidual.data()), this->n, this->n);
    cudaDeviceSynchronize(); // Ensure kernel execution completion

    normalizeQuaternionKernel<<<1, 1>>>(thrust::raw_pointer_cast(this->state_.data()));
    cudaDeviceSynchronize(); // Ensure the normalization completes before proceeding
}

void UKF::updateStateCovariance()
{
    // Step 1: Calculate K * R
    thrust::device_vector<float> KR(this->n * this->n); // Adjust dimensions if necessary
    float* d_KR = thrust::raw_pointer_cast(KR.data());

    cublasSgemm(this->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, this->n, this->n, this->n, &(this->alpha_), thrust::raw_pointer_cast(this->kalmanGain_.data()), this->n, thrust::raw_pointer_cast(this->predictedMeasurementCovariance_.data()), this->n, &(this->beta_), d_KR, this->n);

    // Step 2: Calculate (K * R) * K^T
    thrust::device_vector<float> KRKT(this->n * this->n);
    float* d_KRKT = thrust::raw_pointer_cast(KRKT.data());

    cublasSgemm(this->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, this->n, this->n, this->n, &(this->alpha_), d_KR, this->n, thrust::raw_pointer_cast(this->kalmanGain_.data()), this->n, &(this->beta_), d_KRKT, this->n);

    // Step 3: Update P = P - (K * R) * K^T
    // Negate KRKT for subtraction
    thrust::transform(KRKT.begin(), KRKT.end(), KRKT.begin(), thrust::negate<float>());

    // Add the negated (K * R) * K^T to P
    cublasSaxpy(this->cublasHandle, this->n * this->n, &(this->alpha_), d_KRKT, 1, thrust::raw_pointer_cast(this->P_.data()), 1);
}

void UKF::updateFilter(const std::vector<float>& measurement, const float dt, const float positionVariance, const float orientationVariance, const float velocityVariance, const float angularVelocityVariance, const float linearAccelerationVariance)
{
    this->updateAllSigmaPoints(measurement, dt);
    this->predictStateMean();
    this->predictStateCovariance();
    this->generateMeasurmentSigmaPoints(measurement, dt);
    this->updateMeasurementCovarianceMatrix(positionVariance, orientationVariance, velocityVariance, angularVelocityVariance, linearAccelerationVariance);
    this->predictMeasurementMean();
    this->computeMeasurementCovariance();
    this->computeCrossCovariance();
    this->computeKalmanGain();
    this->updateStateWithMeasurement(measurement, dt);
    this->updateStateCovariance();
}

std::vector<float> UKF::getState()
{
    // Create a standard vector with the same size as the device vector
    std::vector<float> stateHost(this->state_.size());

    // Copy data from device vector to host vector
    thrust::copy(this->state_.begin(), this->state_.end(), stateHost.begin());

    return stateHost;
}

std::vector<float> UKF::getStateCovarianceMatrix()
{
    // Allocate a std::vector<float> with the appropriate size
    std::vector<float> state_covariance_matrix(this->n * this->n);

    // Copy the data from the device_vector to the std::vector
    thrust::copy(this->P_.begin(), this->P_.end(), state_covariance_matrix.begin());

    return state_covariance_matrix;
}

void UKF::printStateInfo() 
{
    std::stringstream ss;

    // State vector
    ss << "State: ";
    for (size_t i = 0; i < this->state_.size(); ++i) {
        ss << this->state_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");
    
    // State Covariance matrix
    ss << "State Covariance matrix: ";
    for (size_t i = 0; i < this->P_.size(); ++i) {
        ss << this->P_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");

    // Process Covariance matrix
    ss << "Process Covariance matrix: ";
    for (size_t i = 0; i < this->Q_.size(); ++i) {
        ss << this->Q_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");

    // Measurement Covariance matrix
    ss << "Measurement Covariance matrix: ";
    for (size_t i = 0; i < this->R_.size(); ++i) {
        ss << this->R_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");

    // Sigma Points matrix
    ss << "Sigma Points matrix: ";
    for (size_t i = 0; i < this->sigmaPoints_.size(); ++i) {
        ss << this->sigmaPoints_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");

    // Weights for sigma points (Mean)
    ss << "Weights for sigma points (Mean): ";
    for (size_t i = 0; i < this->weightsMean_.size(); ++i) {
        ss << this->weightsMean_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");

    // Weights for sigma points (Covariance)
    ss << "Weights for sigma points (Covariance): ";
    for (size_t i = 0; i < this->weightsCovariance_.size(); ++i) {
        ss << this->weightsCovariance_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");
    
    // Predicted State Mean
    ss << "Predicted State Mean: ";
    for (size_t i = 0; i < this->predictedStateMean_.size(); ++i) {
        ss << this->predictedStateMean_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");

    // Predicted State Covariance
    ss << "Predicted State Covariance: ";
    for (size_t i = 0; i < this->predictedStateCovariance_.size(); ++i) {
        ss << this->predictedStateCovariance_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");
    
    // Predicted Measurement Sigma Points
    ss << "Predicted Measurement Sigma Points: ";
    for (size_t i = 0; i < this->predictedMeasurementSigmaPoints_.size(); ++i) {
        ss << this->predictedMeasurementSigmaPoints_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");
    
    // Predicted Measurement Mean
    ss << "Predicted Measurement Mean: ";
    for (size_t i = 0; i < this->predictedMeasurementMean_.size(); ++i) {
        ss << this->predictedMeasurementMean_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");
    
    // Predicted Measurement Covariance
    ss << "Predicted Measurement Covariance: ";
    for (size_t i = 0; i < this->predictedMeasurementCovariance_.size(); ++i) {
        ss << this->predictedMeasurementCovariance_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");
    
    // Cross-Covariance matrix between state and measurement
    ss << "Cross-Covariance matrix between state and measurement: ";
    for (size_t i = 0; i < this->crossCovariance_.size(); ++i) {
        ss << this->crossCovariance_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
    ss.str("");
    
    // Kalman Gain
    ss << "Kalman Gain: ";
    for (size_t i = 0; i < this->kalmanGain_.size(); ++i) {
        ss << this->kalmanGain_[i] << " ";
    }
    RCLCPP_INFO(this->rosLogger_, ss.str().c_str());
}