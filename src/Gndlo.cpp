#include "Gndlo.h"

#include <unistd.h>

using namespace std;
using namespace Eigen;
using namespace cv;


GNDLO::GNDLO()
{
	// Reset execution time
	execution_time = 0.f;
	avg_exec_time = 0.f;

	number_frames = 0;

	rows = 64;
	cols = 1000;

	residual_mad = 5e-5;

	pose_t.setIdentity();

	transform_t.setIdentity();

	num_valid_points = 0;
}

void GNDLO::initialize()
{

	// Initialize the sensor
	initializeSensor();

    number_frames = 0;

	// Set limits to number of threads
	options.num_threads = std::max( std::min(options.num_threads, 8), 1);

	// Initialize levels
	curr_lvl.setResolution(rows, cols);
	prev_lvl.setResolution(rows, cols);

	// Initialize twists and covariance
	twcov.setZero();
	twcov_prev.setZero();

	// Initialize mad
	residual_mad = options.huber_loss;

	// Time breakdown
	avg_time_bd.resize(11);
	avg_time_bd.setZero();
	elapsed_times.resize(11);
	elapsed_times.setZero();

	// Process first frame
	prev_lvl = curr_lvl;
	curr_lvl.d = depth_raw;
	calculateXYZ(curr_lvl);
	selection(curr_lvl, szdata1);

	// Ground initialization
	gnd0.clear();
	gnd1.clear();
		// Initialize ground to (0,0,-1) to start following it
	gnd0.normal << 0., 0., -1.;
	labelPatches(szdata1, &gnd0);
	gnd1 = combineGroundAvg(szdata1);
	gnd0 = gnd1;

	// Show initialization parameters
	if (options.flag_verbose)
	{
		cout << "Initialization complete with the following parameters:" << endl;
		cout << "\tInput resolution: " << cols << "x" << rows << endl;
		cout << "\tNumber of threads: " << options.num_threads << endl;
		cout << "\tValid ratio: " << options.valid_ratio << endl;
		cout << "\tFlags:" << endl;
		cout << "\t\tVerbose: " << options.flag_verbose << endl;
		cout << "\t\tBlur flatness image: " << options.flag_flat_blur << endl;
		cout << "\t\tUse back&forth estimation: " << options.flag_solve_backforth << endl;
		cout << "\t\tUse filter with previous frame: " << options.flag_filter << endl;
		cout << "\tGaussian filtering: " << endl;
		cout << "\t\tKernel radius: " << options.select_radius << endl;
		cout << "\t\tSigma: " << options.gaussian_sigma << endl;
		cout << "\tQuadtrees: " << endl;
		cout << "\t\tAvg. threshold: " << options.quadtrees_avg << endl;
		cout << "\t\tSTD threshold: " << options.quadtrees_std << endl;
		cout << "\t\tMin level: " << options.quadtrees_min_lvl << endl;
		cout << "\t\tMax level: " << options.quadtrees_max_lvl << endl;
		cout << "\tCulling: " << endl;
		cout << "\t\tGoal: " << options.count_goal << endl;
		cout << "\t\tStarting size: " << options.starting_size << endl;
		cout << "\tGround clustering: " << endl;
		cout << "\t\tGround threshold (deg): " << options.ground_threshold_deg << endl;
		cout << "\t\tWall threshold (deg): " << options.wall_threshold_deg << endl;
		cout << "\tSolution: " << endl;
		cout << "\t\tIterations: " << options.iterations << endl;
		cout << "\t\tHuber loss: " << options.huber_loss << endl;
		cout << "\t\tTranslation bound: " << options.trans_bound << endl;
		cout << "\t\tConvergence: " << endl;
		cout << "\t\t\tPixel difference threshold: " << options.pix_threshold << endl;
		cout << "\t\t\tTranslation threshold: " << options.trans_threshold << endl;
		cout << "\t\t\tRotation threshold: " << options.rot_threshold << endl;
	}

}


// -----------------------------------
// MISC
// -----------------------------------
float GNDLO::obtainOrthogonality(const SizedData & szdata)
{
	MatrixXf norm_mat(szdata.normals.size(), 3);
	for (int i=0; i<szdata.normals.size(); ++i)
		norm_mat.row(i) = szdata.normals[i];

	// Determinant
	float orthog = abs((norm_mat.transpose()*norm_mat).determinant()) / pow(szdata.normals.size()/3, 3);

	return orthog;
}


// -----------------------------------
// OVERARCHING FUNCTIONS
// -----------------------------------

void GNDLO::selection(Level & lvl, SizedData & szdata)
{
	unsigned int rr, cc;
	curr_lvl.getResolution(rr, cc);

	// VALIDS AND FLATNESS
	//---------------------------------------------------------
	// Validity of neighborhood
	int number_convolutions = 3;
	int kernel_size = options.select_radius*number_convolutions*2+1;
	cv::Mat validimg(rr, cc, CV_32FC1);
	Map<Matrix<float, Dynamic, Dynamic, RowMajor>> validpixels(validimg.ptr<float>(), rr, cc);
	validpixels = (lvl.d.array() > 0).cast<float>();
	cv::boxFilter(validimg, validimg, -1, Size(kernel_size, kernel_size), Point(-1,-1), true, BORDER_ISOLATED);
	lvl.valids = (validpixels.array() == 1.f);

	// Planarity of range image
	MatrixXf planarity;
	int plan_kernel_size = options.select_radius*2+1;
	obtainPlanarityCurvature(lvl, plan_kernel_size, planarity);

	// Blur the planarity image
	if (options.flag_flat_blur)
	{
		cv::Mat blur_planar(rr, cc, CV_32FC1);
		eigen2cv(planarity, blur_planar);
		cv::GaussianBlur(blur_planar, blur_planar,
							cv::Size(plan_kernel_size, plan_kernel_size),
							options.gaussian_sigma);
		cv2eigen(blur_planar, planarity);
	}

	lvl.planar = planarity;


	// Time elapsed for planarity
	elapsed_time = std::chrono::steady_clock::now();
	diff_time = elapsed_time - prev_time;
	prev_time = elapsed_time;
	elapsed_times(0) = diff_time.count();



	// PLANE SELECTION AND FITTING
	//---------------------------------------------------------
	szdata.clear();

	// Make selection of planes on Z0
	selectQuadtree(lvl, szdata);

	// Time elapsed for quadtrees
	elapsed_time = std::chrono::steady_clock::now();
	diff_time = elapsed_time - prev_time;
	prev_time = elapsed_time;
	elapsed_times(1) = diff_time.count();

	// Obtain centers and normals
	obtainPlanesBlocks(lvl, szdata);

	// Time elapsed for plane fitting
	elapsed_time = std::chrono::steady_clock::now();
	diff_time = elapsed_time - prev_time;
	prev_time = elapsed_time;
	elapsed_times(2) = diff_time.count();
}

void GNDLO::solutionDecoupled(const Level & lvl, SizedData * szdata,
								const Ground * gnd_old, const Ground * gnd_new)
{
	// Our solutions
	TwistCov twcov_loop;
	twcov_loop.twist.setZero();

	// Ceres
	Matrix<double,6,1> ceres_solu;
	Matrix<double,6,6> ceres_cov;
	ceres_cov.setZero();
	ceres_solu = twcov_loop.twist.cast<double>();

	// Convergence
	Matrix<float,6,1> old_twist = twcov_loop.twist;

	// Align ground planes
	TwistCov twcov_align = alignGround(gnd_old, gnd_new);
	Matrix4f ground_T = calculateTransform(twcov_align.twist);

	// Time elapsed for ground alignment
	elapsed_time = std::chrono::steady_clock::now();
	diff_time = elapsed_time - prev_time;
	prev_time = elapsed_time;
	elapsed_times(6) = diff_time.count();

	// Transform patches
	transformPatches(szdata, ground_T.inverse());

	// Loop
	for (int it=0; it<options.iterations; ++it)
	{
		// CORRESPONDING POINTS
		//---------------------------------------------------------
		Matrix4f s_T = calculateTransform(twcov_loop.twist);

		// Transform the points and find them in Z1
		std::vector<Vector2i> old_pixels = szdata->px1;
		transformPixels(szdata->centers, s_T.inverse(), szdata->px1);

		// Check the difference in pixels
		float pixel_difference = 0;
		for (int i=0; i<szdata->px1.size(); ++i)
			pixel_difference += (szdata->px1[i] - old_pixels[i]).norm();

		// Obtain points of the warped coordinates
		obtainPoints(lvl, szdata);

		// Time elapsed for obtaining points
		elapsed_time = std::chrono::steady_clock::now();
		diff_time = elapsed_time - prev_time;
		prev_time = elapsed_time;
		elapsed_times(7) += diff_time.count();


		// MOTION ESTIMATION
		//---------------------------------------------------------
		// Run estimator
		solve2DMotion(szdata, gnd_old, ceres_solu, ceres_cov, residual_mad);

		// Save the solution
		twcov_loop.twist = ceres_solu.cast<float>();
		twcov_loop.cov = ceres_cov.cast<float>();

		// Time elapsed for motion estimation
		elapsed_time = std::chrono::steady_clock::now();
		diff_time = elapsed_time - prev_time;
		prev_time = elapsed_time;
		elapsed_times(8) += diff_time.count();

		// Count iterations:
		elapsed_times(9)++;


		// CONVERGENCE
		//---------------------------------------------------------
		// Pixel difference
		if (pixel_difference < options.pix_threshold)
			break;

		// Translation
		float dist_from_old = (twcov_loop.twist.topRows(3) - old_twist.topRows(3)).norm();
		if (dist_from_old < options.trans_threshold)
			break;

		// Rotation
		MatrixXf old_t = calculateTransform(old_twist);
		MatrixXf new_t = calculateTransform(twcov_loop.twist);
		new_t = new_t.inverse() * old_t;
		float angle_from_old = acosf((new_t.trace()-1)/2);
		if (angle_from_old < options.rot_threshold)
			break;

		// Save the previous results
		old_twist = twcov_loop.twist;
	}

	// Undo transform for next frame
	transformPatches(szdata, ground_T);

	// Combine solutions into global
	twcov.twist = calculateTwist(ground_T * calculateTransform(twcov_loop.twist));
	twcov.cov = twcov_loop.cov + twcov_align.cov;
}

void GNDLO::solutionDecoupledBackForth(const Level & lvl_old, const Level & lvl_new,
								SizedData * szdata_old, SizedData * szdata_new,
								const Ground * gnd_old, const Ground * gnd_new)
{
	// Our solutions
	TwistCov twcov_loop;
	twcov_loop.twist.setZero();

	// Ceres
	Matrix<double,6,1> ceres_solu;
	Matrix<double,6,6> ceres_cov;
	ceres_cov.setZero();
	ceres_solu = twcov_loop.twist.cast<double>();

	// Convergence
	Matrix<float,6,1> old_twist = twcov_loop.twist;

	// Align ground planes
	TwistCov twcov_align = alignGround(gnd_old, gnd_new);
	Matrix4f ground_T = calculateTransform(twcov_align.twist);

	// Time elapsed for ground alignment
	elapsed_time = std::chrono::steady_clock::now();
	diff_time = elapsed_time - prev_time;
	prev_time = elapsed_time;
	elapsed_times(6) = diff_time.count();

	// Transform patches
	transformPatches(szdata_old, ground_T.inverse());
	transformPatches(szdata_new, ground_T);

	// Loop
	for (int it=0; it<options.iterations; ++it)
	{
		// CORRESPONDING POINTS
		//---------------------------------------------------------
		Matrix4f s_T = calculateTransform(twcov_loop.twist);

		// Transform the points and find them in Z1
		std::vector<Vector2i> old_pixels_old = szdata_old->px1;
		std::vector<Vector2i> old_pixels_new = szdata_new->px1;
		transformPixels(szdata_old->centers, s_T.inverse(), szdata_old->px1);
		transformPixels(szdata_new->centers, s_T, szdata_new->px1);

		// Check the difference in pixels
		float pixel_difference = 0;
		for (int i=0; i<szdata_old->px1.size(); ++i)
			pixel_difference += (szdata_old->px1[i] - old_pixels_old[i]).norm();
		for (int i=0; i<szdata_new->px1.size(); ++i)
			pixel_difference += (szdata_new->px1[i] - old_pixels_new[i]).norm();

		// Obtain points of the warped coordinates
		obtainPoints(lvl_new, szdata_old);
		obtainPoints(lvl_old, szdata_new);

		// Time elapsed for obtaining points
		elapsed_time = std::chrono::steady_clock::now();
		diff_time = elapsed_time - prev_time;
		prev_time = elapsed_time;
		elapsed_times(7) += diff_time.count();


		// MOTION ESTIMATION
		//---------------------------------------------------------
		// Run estimator
		solve2DMotionBackForth(szdata_old, szdata_new, gnd_old, ceres_solu, ceres_cov, residual_mad);

		// Save the solution
		twcov_loop.twist = ceres_solu.cast<float>();
		twcov_loop.cov = ceres_cov.cast<float>();

		// Time elapsed for motion estimation
		elapsed_time = std::chrono::steady_clock::now();
		diff_time = elapsed_time - prev_time;
		prev_time = elapsed_time;
		elapsed_times(8) += diff_time.count();

		// Count iterations:
		elapsed_times(9)++;


		// CONVERGENCE
		//---------------------------------------------------------
		float dist_from_old = (twcov_loop.twist.topRows(3) - old_twist.topRows(3)).norm();
		MatrixXf old_t = calculateTransform(old_twist);
		MatrixXf new_t = calculateTransform(twcov_loop.twist);
		new_t = new_t.inverse() * old_t;
		float angle_from_old = acosf((new_t.block<3,3>(0,0).trace()-1)/2);

		// Pixel difference
		if (pixel_difference < options.pix_threshold)
			break;

		// Translation
		if (dist_from_old < options.trans_threshold)
			break;

		// Rotation
		if (angle_from_old < options.rot_threshold)
			break;

		// Save the previous results
		old_twist = twcov_loop.twist;
	}

	// Undo transform for next frame
	transformPatches(szdata_old, ground_T);
	transformPatches(szdata_new, ground_T.inverse());

	// Combine solutions into global
	twcov.twist = calculateTwist(ground_T * calculateTransform(twcov_loop.twist));
	twcov.cov = twcov_loop.cov + twcov_align.cov;
}



// -----------------------------------
// MAIN CODE
// -----------------------------------

// LIDAR
void GNDLO::runOdometry()
{
    //Clock to measure the runtime
	prev_time = std::chrono::steady_clock::now();
	elapsed_times.setZero();

    auto start = std::chrono::steady_clock::now();
	std::chrono::duration<float,std::milli> duration;


	// SET VARIABLES AND INPUT
	//---------------------------------------------------------
	// Set input images
	szdata0 = szdata1;
	prev_lvl = curr_lvl;
	curr_lvl.d = depth_raw;
	calculateXYZ(curr_lvl);

	// Set variable
	unsigned int rr, cc;
	unsigned int valid_points;
	curr_lvl.getResolution(rr, cc);
	twcov_prev = twcov;


	// SELECT PLANAR PATCHES BASED ON PLANARITY
	//---------------------------------------------------------
	selection(curr_lvl, szdata1);


	// CLUSTER PATCHES AND FIND GROUND
	//---------------------------------------------------------
	// Save old ground
	gnd0 = gnd1;

	// Data to use as previous from now on
	Ground * ground_prev = &gnd0;
	SizedData * szdata_prev = &szdata0;

	// Label patches
	labelPatches(szdata1, ground_prev);

	// Time elapsed for Labeling
	elapsed_time = std::chrono::steady_clock::now();
	diff_time = elapsed_time - prev_time;
	prev_time = elapsed_time;
	elapsed_times(3) = diff_time.count();


	// Culling the patches
	orthogCulling(szdata1);

	// Time elapsed for culling
	elapsed_time = std::chrono::steady_clock::now();
	diff_time = elapsed_time - prev_time;
	prev_time = elapsed_time;
	elapsed_times(4) = diff_time.count();


	// Combine patches into ground plane (if there are enough ground patches)
	if (std::count(szdata1.labels.begin(), szdata1.labels.end(), 0))
		gnd1 = combineGroundAvg(szdata1);
	else
		cout << "[WARNING]: No ground patches found. Ground plane not updated." << endl;

	// Time elapsed for clustering
	elapsed_time = std::chrono::steady_clock::now();
	diff_time = elapsed_time - prev_time;
	prev_time = elapsed_time;
	elapsed_times(5) = diff_time.count();



	// MOTION ESTIMATION
	//---------------------------------------------------------
	// Run estimation
	SizedData szdata_curr = szdata1;

	if (options.flag_solve_backforth)
		solutionDecoupledBackForth(prev_lvl, curr_lvl, szdata_prev, &szdata_curr, ground_prev, &gnd1);
	else
		solutionDecoupled(curr_lvl, szdata_prev, ground_prev, &gnd1);


	// ORTHOGONALITY
	//---------------------------------------------------------
	#if 0
	orthogonality = obtainOrthogonality(szdata0);
	cout << "Orthogonality: " << orthogonality << endl;
	#endif


	// FILTER BASED ON COVARIANCE
	//---------------------------------------------------------
	if (options.flag_filter)
	{
		float kd = options.filter_kd * pow(twcov.cov.sum(), options.filter_pd);
		float kf = options.filter_kf * pow(twcov.cov.sum(), options.filter_pf);
		twcov = filterSolution(twcov, kd, kf);
	}

	// Time elapsed for filtering
	elapsed_time = std::chrono::steady_clock::now();
	diff_time = elapsed_time - prev_time;
	prev_time = elapsed_time;
	elapsed_times(10) = diff_time.count();


	// SAVE IN GLOBAL POSE
	//---------------------------------------------------------
	// Check if nan
    if (isnan(twcov.twist(3)))
    {
        std::cout << "ERROR: NaNs in solution. Pose not updated." << endl;
        twcov.setZero();
        transform_t.setIdentity();
    }
	else
	{
		transform_t = calculateTransform(twcov.twist);
	}

	// Calculate total pose wrt the world
    pose_t = pose_t*transform_t;


	// FINISH
	//--------------------------------------------
	//Save runtime
	auto end = std::chrono::steady_clock::now();
	duration = end - start;
	execution_time = duration.count();
	++number_frames;
	avg_exec_time = (avg_exec_time*(number_frames-1) + execution_time)/number_frames;

	// Save breakdown of times
	avg_time_bd = (avg_time_bd*(number_frames-1) + elapsed_times)/number_frames;

}
