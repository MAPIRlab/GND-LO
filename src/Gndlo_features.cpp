#include "Gndlo.h"

using namespace Eigen;
using namespace std;


template <typename Scalar>
Eigen::Matrix<Scalar, -1, -1> GNDLO::createGaussKernel(int size, Scalar sigma)
{
	Matrix<Scalar, -1, 1> kernel(size, 1);
	Matrix<Scalar, -1, -1> kernel2d(size, size);

	if (sigma < 0)
		sigma = 0.3*((size-1)*0.5 - 1) + 0.8;

	float divider = sqrt(2 * M_PI) * sigma;
	for(unsigned int i=0; i<size; ++i)
	{
		float x = float(i) - float(size-1)/2;
		float expval = -(pow(x, 2) / (2 * pow(sigma, 2)));
		kernel(i) = exp(expval) / divider;
	}

	// Normalize
	kernel /= kernel.sum();

	// Turn into 2D filter
	kernel2d = kernel*kernel.transpose();

	return kernel2d;
}

void GNDLO::transformPatches(SizedData * szdata, const Eigen::Matrix4f & T)
{
	for (int i=0; i<szdata->centers.size(); ++i)
	{
		szdata->centers[i] = T.topLeftCorner<3,3>() * szdata->centers[i] + T.topRightCorner<3,1>();
		szdata->normals[i] = T.topLeftCorner<3,3>() * szdata->normals[i];
		szdata->covars[i] = T.topLeftCorner<3,3>() * szdata->covars[i] * T.topLeftCorner<3,3>().transpose();
	}
}

// -----------------------------------
// OBTAIN PLANES
// -----------------------------------

void GNDLO::obtainPlanesBlocks(const Level & lvl, SizedData & szdata)
{
	unsigned int rr, cc;
	lvl.getResolution(rr, cc);

	// Fill every selected pixel
	for (unsigned int i=0; i<szdata.px0.size(); ++i)
	{
		// Location
		int v = szdata.px0[i](0);
		int u = szdata.px0[i](1);

		// Size
		int n_r = szdata.sizes[i];

		// Create Gauss filter
		MatrixXf gker = createGaussKernel(n_r, options.gaussian_sigma);

		// Centers
		Block<const MatrixXf> xN(lvl.x, v, u, n_r, n_r);
		Block<const MatrixXf> yN(lvl.y, v, u, n_r, n_r);
		Block<const MatrixXf> zN(lvl.z, v, u, n_r, n_r);

		// Get the valid pixels
		MatrixXf validmask = lvl.valids.block(v, u, n_r, n_r).cast<float>();
		float validcount = validmask.sum();

		// Check if neighbors are valid
		if ( validcount < (options.valid_ratio*n_r*n_r) )
		{
			szdata.labels[i] = -1;
			szdata.centers[i] = Vector3f::Zero();
			szdata.normals[i] = Vector3f::Zero();
			szdata.fitnesses[i] = 0;
			szdata.covars[i] = Matrix3f::Zero();
			continue;
		}

		// Create arrays of valid points
		MatrixXf diff(3, (int) validcount);
		VectorXf weights((int) validcount);
		int cont = 0;
		for (int uu=0; uu<n_r; ++uu)
			for (int vv=0; vv<n_r; ++vv)
				if ( validmask(vv, uu) > 0 )
				{
					weights(cont) = gker(vv,uu);
					diff(0, cont) = xN(vv,uu);
					diff(1, cont) = yN(vv,uu);
					diff(2, cont) = zN(vv,uu);
					cont++;
				}

		// Normalize weights
		weights /= weights.sum();

		// Center by averaging
		Vector3f center = (diff.array().rowwise() * weights.array().transpose()).rowwise().sum();
		szdata.centers[i] = center;

		// Matrix of differences
		diff = diff.colwise() - center;

		// Covariance matrix
		Matrix3f cov = (diff * weights.asDiagonal() * diff.transpose()) / (1. - weights.cwiseAbs2().sum());

		// Normal vector and fitness value from eigen decomposition
		Vector3f ncross;
		float eigval;
		// SVD of the difference of points
		diff = diff.array().rowwise() * weights.array().sqrt().transpose() / sqrt(1. - weights.cwiseAbs2().sum());
		JacobiSVD<MatrixXf> svd(diff, ComputeThinU | ComputeThinV);
		ncross = svd.matrixU().rightCols<1>();
		eigval = svd.singularValues()(2);

		// Check normal direction
		if (ncross.dot(center) < 0)
			ncross = -ncross;

		ncross.normalize();

		// Save normal in data structure
		szdata.covars[i] = cov;
		szdata.normals[i] = ncross;
		szdata.fitnesses[i] = eigval;

	}
}

// -----------------------------------
// OBTAIN POINTS
// -----------------------------------

void GNDLO::obtainPoints(const Level & lvl, SizedData * szdata)
{
	// Initialize
	unsigned int rr, cc;
	int ker_r = options.select_radius;
	int ker_size = ker_r*2+1;
	MatrixXf kernel2d(ker_size, ker_size);
	lvl.getResolution(rr,cc);

	// Create Gaussian filter
	kernel2d = createGaussKernel(ker_size, options.gaussian_sigma);

	// Obtain point
	for (int i=0; i<szdata->px1.size(); ++i)
	{
		// Grab pixel
		int bsize = szdata->sizes[i];
		int v = szdata->px1[i](0);
		int u = szdata->px1[i](1);

		// Check if pixel is valid
		if ( v<ker_r || u<ker_r || v>rr-ker_r || u>cc-ker_r )
		{
			szdata->labels[i] = -1;
			szdata->points[i] = Vector3f::Zero();
			continue;
		}
		if (lvl.d(v,u) == 0.)
		{
			szdata->labels[i] = -1;
			szdata->points[i] = Vector3f::Zero();
			continue;
		}

		// Block of the coordinates
		MatrixXf xn = lvl.x.block(v-ker_r, u-ker_r, ker_size, ker_size);
		MatrixXf yn = lvl.y.block(v-ker_r, u-ker_r, ker_size, ker_size);
		MatrixXf zn = lvl.z.block(v-ker_r, u-ker_r, ker_size, ker_size);
		MatrixXf dn = lvl.d.block(v-ker_r, u-ker_r, ker_size, ker_size);
		ArrayXXf validmask = (dn.array()>0).cast<float>();
		float validcount = validmask.sum();

		// Check if neighbors are valid
		if ( validcount < (options.valid_ratio*ker_size*ker_size) )
		{
			szdata->labels[i] = -1;
			szdata->points[i] = Vector3f::Zero();
			continue;
		}

		// Use the mask of valid pixels
		validmask = validmask * kernel2d.array();
		validcount = validmask.sum();

		// Obtain the averaged coordinates
		float xf = (xn.array() * validmask).sum()/validcount;
		float yf = (yn.array() * validmask).sum()/validcount;
		float zf = (zn.array() * validmask).sum()/validcount;

		szdata->points[i] << xf, yf, zf;

	}
}
