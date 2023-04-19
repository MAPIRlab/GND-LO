#include "Gndlo.h"

#include <math.h>

// Ceres
#include "ceres/ceres.h"


using namespace Eigen;

template<typename T>
T getMad(std::vector<T> v)
{
    std::sort(v.begin(), v.end());
    double res_med = v[int(v.size()/2)];

    for (int i=0; i<v.size(); ++i)
    {
        v[i] = std::abs(std::abs(res_med) - std::abs(v[i]));
    }
    std::sort(v.begin(), v.end());
    double mad = v[int(v.size()/2)];
    return mad;
}


// -----------------------------------
// COVARIANCES
// -----------------------------------
Matrix<double,6,6> calcCovariance2D(const double * trans,
                    const double * theta,
					const MatrixXd & jac,
                    ceres::Problem * problem)
{
	Matrix<double,6,6> cov;

	// Set covariance blocks
	ceres::Covariance::Options cov_options;
	ceres::Covariance ceres_cov(cov_options);
	std::vector<std::pair<const double*, const double*> > covariance_blocks;
	covariance_blocks.push_back(std::make_pair(trans, trans));
	covariance_blocks.push_back(std::make_pair(theta, theta));
	covariance_blocks.push_back(std::make_pair(trans, theta));

	// Compute covariance
	CHECK(ceres_cov.Compute(covariance_blocks, problem));

	// Retrieve covariance matrices
	Matrix2d cov_trans;
	double cov_theta[1];
	MatrixXd cov_mix(2,1);
	ceres_cov.GetCovarianceBlock(trans, trans, cov_trans.data());
	ceres_cov.GetCovarianceBlock(theta, theta, cov_theta);
	ceres_cov.GetCovarianceBlock(trans, theta, cov_mix.data());

	// Compose covariance matrix
	Matrix3d cov_total = Matrix3d::Zero();
	cov_total << cov_trans, cov_mix, cov_mix.transpose(), cov_theta[0];

	// Transform to 6x6
	cov = jac * cov_total * jac.transpose();

    return cov;
}


// -----------------------------------
// FILTER
// -----------------------------------
TwistCov GNDLO::filterSolution(const TwistCov & twcov,
								float cf,
								float df)
{
	// TODO: Check covariance and old_covariance for filter

	TwistCov out;

		// Calculate eigenvalues and eigenvectors
    SelfAdjointEigenSolver<MatrixXf> eigensolver(twcov.cov);
    if (eigensolver.info() != Success)
    {
        std::cout << "Covariance solver error\n";
        std::cout << twcov.twist.transpose() << "\n";
		std::cout << twcov.cov << std::endl;

		return twcov;
    }
	else
	{
		// Describe new twist from eigenvectors basis
	    Matrix<float,6,6> Bii;
	    Matrix<float,6,1> kai_b, aux_solu;
	    Bii = eigensolver.eigenvectors();
	    kai_b = Bii.colPivHouseholderQr().solve(twcov.twist);

		// Describe old twist from eigenvectors basis
	    Matrix<float,6,1> kai_b_old;
	    kai_b_old = Bii.colPivHouseholderQr().solve(twcov_prev.twist);

		// Filter
	    Matrix<float,6,1> kai_b_fil;
	    for (unsigned int i=0; i<6; i++)
		{
			float eigval = eigensolver.eigenvalues()(i,0);
			float fact = pow(eigval, 2);
	        kai_b_fil(i) = (kai_b(i) + (cf*fact + df)*kai_b_old(i))/(1.f + cf*fact + df);
		}

	    //Transform filtered velocity to the local reference frame
	    out.twist = Bii.inverse().colPivHouseholderQr().solve(kai_b_fil);
		out.cov = twcov.cov;
	}

	return out;
}


// -----------------------------------
// CERES SOLVER
// -----------------------------------
void GNDLO::solve2DMotion(const SizedData * indata,
							const Ground * ground,
                            Eigen::Matrix<double,6,1> & twist,
                            Eigen::Matrix<double,6,6> & cov,
                            const double & huber_factor)
{
    int valid_points = indata->centers.size();

    // Create optimization problem
    ceres::Problem problem;
    Vector3d point, center, normal;
	Vector3d pos;
	double theta[1] = {0.};
	double trans[2] = {0., 0.};

	// Get 2D motion translation and rotation axes
		// Rotation axis = ground normal
	Vector3d axis_rot = ground->normal.cast<double>();
		// Translation along (mostly) X
	Vector3d axis_tx = Vector3d::Zero();
	axis_tx(0) = -ground->normal(2);	// Assuming n_z is the highest value of n, and not 0
	axis_tx(2) = ground->normal(0);
	axis_tx.normalize();
		// Translation along the remaining axis
	Vector3d axis_ty = Vector3d::Zero();
	axis_ty = axis_tx.cross(axis_rot);

    // Loss function: Huber
    ceres::LossFunction* loss = new ceres::HuberLoss(huber_factor);

    // Cost functions
    for (int i=0; i<valid_points; ++i)
    {
		// Check if it is a wall patch
		if (indata->labels[i] > 0)
		{
			// Fill variables
			double w = 1.d/(0.5d + indata->fitnesses[i]);

			center = indata->centers[i].cast<double>();
			normal = indata->normals[i].cast<double>();
			point = indata->points[i].cast<double>();

			// Fill Ceres problem
			ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<Motion2DCostFunc, 1, 1, 2>(
				new Motion2DCostFunc(point, center, w*normal, axis_tx, axis_ty, axis_rot));

				problem.AddResidualBlock(cost_function, loss, theta, trans);

		}
    }

	// Check there are enough residual blocks
	if (problem.NumResiduals() < 5)
	{
		std::cout << "Not enough cost functions to estimate motion." << std::endl;
        twist.setZero();
        cov.setZero();
        residual_mad = options.huber_loss;
        return;
	}

	// Solution bounds
	problem.SetParameterLowerBound(trans, 0, -options.trans_bound);
	problem.SetParameterLowerBound(trans, 1, -options.trans_bound);
	problem.SetParameterUpperBound(trans, 0, options.trans_bound);
	problem.SetParameterUpperBound(trans, 1, options.trans_bound);

    // Problem options
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = 1000;
    ceres_options.linear_solver_type = ceres::DENSE_QR;
    ceres_options.minimizer_progress_to_stdout = false;
    ceres_options.num_threads = options.num_threads;
    ceres::Solver::Summary summary;
    ceres::Solve(ceres_options, &problem, &summary);

    // Obtain MAD of residuals for Huber next iteration
    ceres::Problem::EvaluateOptions evalopts;
    evalopts.apply_loss_function = false;
    std::vector<double> residuals;
    problem.Evaluate(evalopts, nullptr, &residuals, nullptr, nullptr);
    residual_mad = getMad(residuals);

    // Output twist
    if (!summary.IsSolutionUsable())
    {
		std::cout << "Solution is not usable." << std::endl;
        twist.setZero();
        cov.setZero();
        residual_mad = options.huber_loss;
        return;
    }

	// Set output twist
	pos = trans[0]*axis_tx + trans[1]*axis_ty;
	twist << pos, theta[0]*axis_rot;

    // Calculate the covariance
	MatrixXd jac(6,3);
	jac << axis_tx, axis_ty, Vector3d::Zero(),
			Vector3d::Zero(), Vector3d::Zero(), axis_rot;
	cov = calcCovariance2D(trans, theta, jac, &problem);

}

void GNDLO::solve2DMotionBackForth(const SizedData * indata_old,
									const SizedData * indata_new,
									const Ground * ground,
		                            Eigen::Matrix<double,6,1> & twist,
		                            Eigen::Matrix<double,6,6> & cov,
		                            const double & huber_factor)
{
    // Create optimization problem
    ceres::Problem problem;
    Vector3d point, center, normal;
	Vector3d pos;
	double theta[1] = {0.};
	double trans[2] = {0., 0.};

	// Get 2D motion translation and rotation axes
		// Rotation axis = ground normal
	Vector3d axis_rot = ground->normal.cast<double>();
		// Translation along (mostly) X
	Vector3d axis_tx = Vector3d::Zero();
	axis_tx(0) = -ground->normal(2);	// Assuming n_z is the highest value of n, and not 0
	axis_tx(2) = ground->normal(0);
	axis_tx.normalize();
		// Translation along the remaining axis
	Vector3d axis_ty = Vector3d::Zero();
	axis_ty = axis_tx.cross(axis_rot);

    // Loss function: Huber
    ceres::LossFunction* loss = new ceres::HuberLoss(huber_factor);

    // Cost functions from old data (S0 planes - S1 points)
	int valid_points = indata_old->centers.size();
    for (int i=0; i<valid_points; ++i)
		if (indata_old->labels[i] > 0)		// Check if it is a wall patch
		{
			// Set weight
			double w = 1.d/(0.5d + indata_old->fitnesses[i]);

			// Fill Ceres problem
			ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<Motion2DCostFunc, 1, 1, 2>(
				new Motion2DCostFunc(indata_old->points[i].cast<double>(),
									indata_old->centers[i].cast<double>(),
									w*indata_old->normals[i].cast<double>(),
									axis_tx, axis_ty, axis_rot));

				problem.AddResidualBlock(cost_function, loss, theta, trans);

		}

		// Cost functions from new data (S1 planes - S0 points)
	valid_points = indata_new->centers.size();
    for (int i=0; i<valid_points; ++i)
		if (indata_new->labels[i] > 0)		// Check if it is a wall patch
		{
			// Set weight
			double w = 1.d/(0.5d + indata_new->fitnesses[i]);

			center = indata_new->centers[i].cast<double>();
			normal = indata_new->normals[i].cast<double>();
			point = indata_new->points[i].cast<double>();

			// Fill Ceres problem
			ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<Motion2DInvCostFunc, 1, 1, 2>(
				new Motion2DInvCostFunc(indata_new->points[i].cast<double>(),
										indata_new->centers[i].cast<double>(),
										w*indata_new->normals[i].cast<double>(),
										axis_tx, axis_ty, axis_rot));

				problem.AddResidualBlock(cost_function, loss, theta, trans);
		}

	// Check there are enough residual blocks
	if (problem.NumResiduals() < 5)
	{
		std::cout << "Not enough cost functions to estimate motion." << std::endl;
        twist.setZero();
        cov.setZero();
        residual_mad = options.huber_loss;
        return;
	}

	// Solution bounds
	problem.SetParameterLowerBound(trans, 0, -options.trans_bound);
	problem.SetParameterLowerBound(trans, 1, -options.trans_bound);
	problem.SetParameterUpperBound(trans, 0, options.trans_bound);
	problem.SetParameterUpperBound(trans, 1, options.trans_bound);

    // Problem options
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = 1000;
    ceres_options.linear_solver_type = ceres::DENSE_QR;
    ceres_options.minimizer_progress_to_stdout = false;
    ceres_options.num_threads = options.num_threads;
    ceres::Solver::Summary summary;
    ceres::Solve(ceres_options, &problem, &summary);

    // Obtain MAD of residuals for Huber next iteration
    ceres::Problem::EvaluateOptions evalopts;
    evalopts.apply_loss_function = false;
    std::vector<double> residuals;
    problem.Evaluate(evalopts, nullptr, &residuals, nullptr, nullptr);
    residual_mad = getMad(residuals);

    // Output twist
    if (!summary.IsSolutionUsable())
    {
		std::cout << "Solution is not usable." << std::endl;
        twist.setZero();
        cov.setZero();
        residual_mad = options.huber_loss;
        return;
    }

	// Set output twist
	pos = trans[0]*axis_tx + trans[1]*axis_ty;
	twist << pos, theta[0]*axis_rot;

    // Calculate the covariance
	MatrixXd jac(6,3);
	jac << axis_tx, axis_ty, Vector3d::Zero(),
			Vector3d::Zero(), Vector3d::Zero(), axis_rot;
	cov = calcCovariance2D(trans, theta, jac, &problem);

}
