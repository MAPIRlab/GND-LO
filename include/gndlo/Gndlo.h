
#ifndef GNDLO_H
#define GNDLO_H

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

// OPENCV
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <chrono>

// Other files
#include "TypeDefs.h"
#include "Ceres_types.h"


//-----------------------------------------

class  GNDLO {
public:

    // Options struct
    // -----------------------------------------
    struct Options {
			// General
		int num_threads = 8;					// For Ceres solver
		float valid_ratio = 0.8;				// Min. ratio of valid pixels in neighborhood to consider it valid
			// Flags for behaviour
		bool flag_verbose = true;				// Display info
		bool flag_flat_blur = true;				// Flag to blur the flatness image
		bool flag_solve_backforth = true;		// Flag to solve using patches from both images
		bool flag_filter = true;				// Flag to filter estimation
			// Image
        int rows = 64;							// Resolution: rows
        int cols = 1000;						// Resolution: cols
			// Gaussian filtering
		int select_radius = 1;					// Gaussian kernel: radius
		float gaussian_sigma = -1;				// Gaussian kernel: sigma
			// Quadtree-based selection
		float quadtrees_avg = 0.1;				// Quadtree selection: average threshold to select patch
		float quadtrees_std = 0.015;			// Quadtree selection: STD threshold to divide block
		float quadtrees_min_lvl = 2;			// Quadtree selection: finest level (2^quadtrees_min_lvl)
		float quadtrees_max_lvl = 5;			// Quadtree selection: coarsest level (2^quadtrees_max_lvl)
			// Orthogonality culling
		float count_goal = 50;					// Culling: number of patches that should contribute to every direction
		int starting_size = 4;					// Culling: initialize with patches of this size and higher
			// Ground clustering
		float ground_threshold_deg = 10.;		// Labeling: angle difference threshold to consider a patch as ground
		float wall_threshold_deg = 60.;			// Labeling: angle difference threshold to consider a patch as wall
			// Solution parameters
		int iterations = 20;					// Max iterations of the algorithm
		float huber_loss = 3e-5;				// Starting huber loss
		double trans_bound = 1.;				// Limits for the translation (in m)
			// Covergence parameters
		float pix_threshold = 5.;				// Convergence: pixel difference threshold
		float trans_threshold = 0.;				// Convergence: translation norm threshold
		float rot_threshold = 0.;				// Convergence: rotation angle threshold
			// Filter parameters
		float filter_kd = 100.;
		float filter_pd = 0.;
		float filter_kf = 2.;
		float filter_pf = 1.;
			// Output
		bool flag_save_results = false;			// Flag to save the results in Freiburg style
		std::string results_file_name = "";		// Where to save the results
    };

protected:

	 // Options
	Options options;

	 // Timing variables
	Eigen::ArrayXf elapsed_times;
	std::chrono::time_point<std::chrono::steady_clock> prev_time;
	std::chrono::time_point<std::chrono::steady_clock> elapsed_time;
	std::chrono::duration<float,std::milli> diff_time;

     // Matrix that stores the original frames with the image resolution
    Eigen::MatrixXf depth_raw;

	 // Levels used
	Level curr_lvl;
	Level prev_lvl;

	 // Structure of patches and corresponding points
	SizedData szdata0, szdata1;

	 // Ground in both frames
	Ground gnd0, gnd1;

	 // Solution and covariance
	TwistCov twcov;

	 // Solution and covariance of previous frame
	TwistCov twcov_prev;

     // Huber loss for Ceres
    double residual_mad;

     // Auxiliar matrix
    Eigen::MatrixXf auxmat;

     // Least squares covariance matrix
    Eigen::Matrix<float, 6, 6> est_cov;

     // Number of frames calculated
    unsigned long int number_frames;

     // The resolution of the input image
    unsigned int rows;
    unsigned int cols;

     // Total transformation matrix
    Eigen::Matrix4f transform_t;

     // World pose transformation matrix
    Eigen::Matrix4f pose_t;


	// ------------------------------------
	// LIDAR.cpp
	// ------------------------------------
     // Obtain XYZ coordinates from depth image
    virtual void calculateXYZ(Level & lvl) {}

	 // Transform pixels using the optical flow from a transformation
	virtual void transformPixels(const std::vector<Eigen::Vector3f> & centers,
									const Eigen::Matrix4f & transform,
									std::vector<Eigen::Vector2i> & outpixels) {}


	// ------------------------------------
	// SOLUTION.cpp
	// ------------------------------------
	 // Estimate the 2D motion based on point-plane pairs
	void solve2DMotion(const SizedData * indata,
						const Ground * ground,
						Eigen::Matrix<double,6,1> & twist,
						Eigen::Matrix<double,6,6> & cov,
						const double & huber_factor);

	 // Estimate the 2D motion based on point-plane pairs, back and forth
	void solve2DMotionBackForth(const SizedData * indata_old,
						const SizedData * indata_new,
						const Ground * ground,
                        Eigen::Matrix<double,6,1> & twist,
                        Eigen::Matrix<double,6,6> & cov,
                        const double & huber_factor);

     // Returns the twist after combining it with the old twist using the covariance of the new solution
    TwistCov filterSolution(const TwistCov & twcov,
							float cf,
							float df);


	// ------------------------------------
	// PLANARITY.cpp
	// ------------------------------------
     // Obtain planarity of a level using the curvature
    void obtainPlanarityCurvature(const Level & lvl, int ksize, Eigen::MatrixXf & out);


	// ------------------------------------
	// SELECTION.cpp
	// ------------------------------------
	 // Select based on quadtrees using thresholds
	void selectQuadtree(const Level & lvl, SizedData & outdata);
	 // Select based on quadtrees using thresholds
	void orthogCulling(SizedData & data);


	// ------------------------------------
	// FEATURES.cpp
	// ------------------------------------
	 // Create Gaussian kernel for filtering
	template <typename Scalar>
	Eigen::Matrix<Scalar, -1, -1> createGaussKernel(int size, Scalar sigma);
	 // Transform patches
	void transformPatches(SizedData * szdata, const Eigen::Matrix4f & T);
     // Fill the data structure with centers and normals using blocks (SVD)
    void obtainPlanesBlocks(const Level & lvl, SizedData & szdata);
     // Fill the data structure with the points associated to each plane
    void obtainPoints(const Level & lvl, SizedData * szdata);


	// ------------------------------------
	// GROUND.cpp
	// ------------------------------------
	 // Combine patches into ground plane using average
	Ground combineGroundAvg(const SizedData & szdata);
	 // Cluster the ground plane based on simple threshold
	void labelPatches(SizedData & szdata, const Ground * old_gnd);
	 // Find transformation that align two ground planes
	TwistCov alignGround(const Ground * gnd_old, const Ground * gnd_new);


	// ------------------------------------
	// Gndlo.cpp
	// ------------------------------------
	 // Obtain orthogonality of selected patches
	float obtainOrthogonality(const SizedData & szdata);
	 // Do preprocessing and selection
	void selection(Level & lvl, SizedData & szdata);
	 // Estimate the motion by aligning ground planes, then 2D motion
	void solutionDecoupled(const Level & lvl, SizedData * szdata, const Ground * gnd_old, const Ground * gnd_new);
	 // Estimate the motion by aligning ground planes, then 2D motion, using both sets of patches
	void solutionDecoupledBackForth(const Level & lvl_old, const Level & lvl_new,
									SizedData * szdata_old, SizedData * szdata_new,
									const Ground * gnd_old, const Ground * gnd_new);


public:

     // Num of valid points after removing null pixels
    unsigned int num_valid_points;

     // Execution time (ms)
    float execution_time;
    float avg_exec_time;
	Eigen::ArrayXf avg_time_bd;

     // Selected pixels
    float orthogonality;

	 // Run odometry
    void runOdometry();

	 // Degrees to radians
	template <typename Derived>
	Derived deg2rad(const Derived in)
	{
		Derived out = in * Derived(M_PI) / Derived(180.);
		return out;
	}

	 // Radians to degrees
	template <typename Derived>
	Derived rad2deg(const Derived in)
	{
		Derived out = in * Derived(180.) / Derived(M_PI);
		return out;
	}

	 // Calculate hat matrix
	template <typename Derived>
	Eigen::Matrix<typename Derived::Scalar, 3, 3> hatMatrix(const Eigen::MatrixBase<Derived>& in)
	{
		// Create output matrix
		Eigen::Matrix<typename Derived::Scalar, 3, 3> out;

		// Check right size of vector
		if (in.size() != 3)
		{
			std::cout << "Cannot hat operator matrix of " << in.rows() << "x" << in.cols() << std::endl;
			out.setZero();
			return out;
		}

		// Create and output
		out << 0., -in(2), in(1),
				in(2), 0., -in(0),
				-in(1), in(0), 0.;
		return out;
	}

     // Calculate T matrix from twist vector
    template <typename Derived>
    Eigen::Matrix<typename Derived::Scalar, -1, -1> calculateTransform(const Eigen::MatrixBase<Derived>& chi)
    {
        Eigen::Matrix<typename Derived::Scalar, 4, 4> T, Tchi;
        Tchi.setZero();
        Tchi(0,1) = -chi(5); Tchi(0,2) =  chi(4);
        Tchi(1,0) =  chi(5); Tchi(1,2) = -chi(3);
        Tchi(2,0) = -chi(4); Tchi(2,1) =  chi(3);

        #if 1
        // Pseudo exponential
        T = Tchi.exp();
        T(0,3) = chi(0); T(1,3) = chi(1); T(2,3) = chi(2);
        #else
        // Real exponential
        Tchi(0,3) = chi(0); Tchi(1,3) = chi(1); Tchi(2,3) = chi(2);
        T = Tchi.exp();
        #endif

        return T;
    }

     // Calculate twist vector from T matrix
    template <typename Derived>
    Eigen::Matrix<typename Derived::Scalar, -1, -1> calculateTwist(const Eigen::MatrixBase<Derived>& T)
    {
        Eigen::Matrix<typename Derived::Scalar, 6, 1> chi;
        Eigen::Matrix<typename Derived::Scalar, 4, 4> Tchi;
        Tchi = T.log();

        #if 1
        // Pseudo logarithm
        chi(0) = T(0,3);    chi(1) = T(1,3);    chi(2) = T(2,3);
        chi(3) = Tchi(2,1); chi(4) = Tchi(0,2); chi(5) = Tchi(1,0);
        #else
        // Real logarithm
        chi(0) = Tchi(0,3);    chi(1) = Tchi(1,3);    chi(2) = Tchi(2,3);
        chi(3) = Tchi(2,1);    chi(4) = Tchi(0,2);    chi(5) = Tchi(1,0);
        #endif

        return chi;
    }

	 // Pseudo inverse of square matrix by inversing eigenvalues above a threshold
	template <typename Derived>
	Eigen::Matrix<typename Derived::Scalar, -1, -1> pseudoInverse(const Eigen::MatrixBase<Derived>& in, const float threshold)
	{
		// Check shape
		if (in.rows() != in.cols())
		{
			std::cout << "WARNING: pseudoInverse can only be applied to square matrices.";
			return in;
		}

		// Eigen decomposition
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(in);

		// Invert eigenvalues
		Eigen::VectorXf vals(in.rows());
		for (int i=0; i<in.rows(); ++i)
			vals(i) = (eig.eigenvalues()(i) > threshold) ? 1./eig.eigenvalues()(i) : 0;

		// Create output matrix
		Eigen::Matrix<typename Derived::Scalar, -1, -1> out;
		out = eig.eigenvectors().transpose() * vals.asDiagonal() * eig.eigenvectors();

		return out;
	}


     // Depth input
    inline void setDepthInput(Eigen::MatrixXf &d) {depth_raw = d;}

     // Initialize parameters
    void initialize(void);

     // Specific initialization based on sensor
    virtual void initializeSensor(void) {}

     // Get frame number
    inline unsigned long int getFrameNumber() {return number_frames;}

     // Get odometry result
    inline void getOdoResult(Eigen::VectorXf &odores) {odores = twcov.twist;}
     // Get covariance of the solution
    inline void getCovariance(Eigen::MatrixXf &outcov) {outcov = twcov.cov;}
     // Get total pose wrt world
    inline void getPose(Eigen::Matrix4f &pose) {pose = pose_t;}


    //Constructor. Initialize variables and matrix sizes
    GNDLO();

};

#endif
