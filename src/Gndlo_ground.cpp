#include "Gndlo.h"

using namespace Eigen;


// -----------------------------------
// COMBINE PATCHES INTO GROUND
// -----------------------------------

Ground GNDLO::combineGroundAvg(const SizedData & szdata)
{
	// Declare output structure
	Ground gnd;

	// Check number of ground patches
	int n_ground_patches = 0;
	for (int i=0; i<szdata.labels.size(); ++i)
		if (szdata.labels[i] == 0)
			n_ground_patches++;

	// Fill matrices with ground patches
	int j = 0;
	MatrixXf Mc(3, n_ground_patches), Mn(3, n_ground_patches);
	VectorXf Mw(n_ground_patches);
	for (int i=0; i<szdata.labels.size(); ++i)
	{
		if (szdata.labels[i] == 0)
		{
			// Fill matrices
			Mw(j) = pow(szdata.sizes[i], 2);
			Mc.col(j) = szdata.centers[i];
			Mn.col(j) = szdata.normals[i];

			// Update counter
			++j;
		}
	}

	// Initialize structure
	gnd.clear();
	gnd.count_patches = n_ground_patches;
	gnd.count_px = Mw.sum();

	// Normalize weights
	Mw /= Mw.sum();

	// Average values
	gnd.center = (Mc.array().rowwise() * Mw.transpose().array()).rowwise().sum();
	gnd.normal = (Mn.array().rowwise() * Mw.transpose().array()).rowwise().sum();
	gnd.normal.normalize();

	// Set matrices
	Mc.array().colwise() -= gnd.center.array();
	Mn.array().colwise() -= gnd.normal.array();

	// Covariance matrix
	gnd.covariance << (Mc * Mw.asDiagonal() * Mc.transpose()),
					  (Mc * Mw.asDiagonal() * Mn.transpose()),
					  (Mn * Mw.asDiagonal() * Mc.transpose()),
					  (Mn * Mw.asDiagonal() * Mn.transpose());

	// Output
	return gnd;
}


// -----------------------------------
// GROUND OPERATIONS
// -----------------------------------

void GNDLO::labelPatches(SizedData & szdata, const Ground * old_gnd)
{
	// Ground patches are around PI because of acos(x)
	float gnd_cos_threshold = cos(deg2rad(options.ground_threshold_deg));
	float wall_cos_threshold = cos(M_PI/2.f - deg2rad(options.wall_threshold_deg));

	// Cluster based on threshold on beta = acos(nz)
	int n_ground_patches = 0;
	int n_wall_patches = 0;
	for (int i=0; i<szdata.normals.size(); ++i)
	{
		float cos_angle = szdata.normals[i].dot(old_gnd->normal);
			// Check ground
		if (cos_angle > gnd_cos_threshold)
		{
			szdata.labels[i] = 0;
			n_ground_patches++;
		}
			// Check wall
		else if (abs(cos_angle) < wall_cos_threshold)
		{
			szdata.labels[i] = 1;
			n_wall_patches++;
		}
			// Set to outlier
		else
		{
			szdata.labels[i] = -1;
		}
	}

}

TwistCov GNDLO::alignGround(const Ground * gnd_old, const Ground * gnd_new)
{
	// Create output structure
	TwistCov out;

	// Find T that aligns both ground planes
	Vector3f crossproduct = gnd_new->normal.cross(gnd_old->normal);
	float distance = gnd_old->normal.dot(gnd_old->center) - gnd_new->normal.dot(gnd_new->center);
	Vector3f translation = gnd_old->normal * distance;

	// Fill twist
	out.twist << translation, crossproduct;

	// Fill jacobians wrt to both input grounds
	MatrixXf F_tw_old(6,6), F_tw_new(6,6);

		// Twist to ground old
	F_tw_old << gnd_old->normal * gnd_old->normal.transpose(),
				gnd_old->normal * gnd_old->center.transpose() + Matrix3f::Identity()*distance,
				Matrix3f::Zero(),
				hatMatrix(gnd_new->normal);

		// Twist to ground new
	F_tw_new << -gnd_old->normal * gnd_new->normal.transpose(),
				-gnd_old->normal * gnd_new->center.transpose(),
				Matrix3f::Zero(),
				-hatMatrix(gnd_old->normal);

	// Calculate covariance of output by propagation
	out.cov = F_tw_old*gnd_old->covariance*F_tw_old.transpose()
			+ F_tw_new*gnd_new->covariance*F_tw_new.transpose();

	return out;
}
