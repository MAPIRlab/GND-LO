#include "Gndlo.h"

#include <numeric>

using namespace Eigen;
using namespace std;


void GNDLO::selectQuadtree(const Level & lvl, SizedData & outdata)
{
	// Parameters
	int max_lvl = options.quadtrees_max_lvl;
	int min_lvl = options.quadtrees_min_lvl;
	float std_th = options.quadtrees_std;
	float avg_th = options.quadtrees_avg;
	float valid_ratio = options.valid_ratio;

	// Initialization
	int rr, cc;
	curr_lvl.getResolution(rr, cc);
	MatrixXf group_size = MatrixXf::Zero(rr,cc);

	// Go through every block size (pyramid style)
	float avg, std;
	for (int i=max_lvl; i>=min_lvl; --i)
	{
		int block_size = pow(2, i);		// Calculate block size

		// Go over image block by block
		for (int u=0; u<cc; u+=block_size)
		{
			int usize = std::min(block_size, int(cc)-u);

			for (int v=0; v<rr; v+=block_size)
			{
				int vsize = std::min(block_size, int(rr)-v);

				// Grab the block
				Block<const MatrixXf> block(lvl.planar, v, u, vsize, usize);

				// Use valid mask
				MatrixXf validmask = lvl.valids.block(v, u, vsize, usize).cast<float>();
				float validcount = validmask.sum();

				// Check minimum number of pixels
				if (validcount < (valid_ratio*vsize*usize))
					continue;

				// Calculate the average
				avg = (block.array() * validmask.array()).sum()/validcount;

				// Calculate STD
				MatrixXf diff = block.array() - avg;
				std = sqrt( (diff.array() * validmask.array()).abs2().sum()/validcount );

				// Check to group
				if (std<(std_th) && group_size(v,u)==0)
				{
					group_size.block(v, u, vsize, usize) = MatrixXf::Constant(vsize, usize, block_size);

					// Check to add to selection
					if (avg < avg_th)
					{
						outdata.px0.push_back(Vector2i(v,u));
						outdata.px1.push_back(Vector2i(v+(block_size/2), u+(block_size/2)));
						outdata.covars.push_back(Matrix3f::Zero());
						outdata.normals.push_back(Vector3f::Zero());
						outdata.centers.push_back(Vector3f::Zero());
						outdata.fitnesses.push_back(avg);
						outdata.sizes.push_back(block_size);
						outdata.labels.push_back(0);
						outdata.points.push_back(Vector3f::Zero());
					}
				}
			}
		}
	}
}

void GNDLO::orthogCulling(SizedData & data)
{
	// Erase outliers (not labelled)
	for (int i=data.labels.size()-1; i>=0; --i)
		if (data.labels[i] < 0)
			data.erase(i);

	// Obtain orthogonality from sizes starting_size and up
	Matrix3f norm_matrix = Matrix3f::Zero();
	for (int i=0; i<data.normals.size(); ++i)
		if (data.sizes[i] >= options.starting_size)
			norm_matrix += (data.normals[i] * data.normals[i].transpose());

	// Go through sizes
	for (int sz=options.starting_size/2; sz>=2; sz/=2)
	{
		// Eigen
		SelfAdjointEigenSolver<Matrix3f> eig(norm_matrix);

		// Check for directions with not enough patches
		Vector3f diff_from_goal = options.count_goal - eig.eigenvalues().array();
		Vector3i diff_bool = (diff_from_goal.array() > 0).cast<int>();

		// Check for patches that fill this void
		for (int i=0; i<data.labels.size(); ++i)
		{
			if (data.sizes[i] == sz)
			{
				// Angular error with every eigen vector
				Vector3f angle_diff;
				for (int j=0; j<3; ++j)
					angle_diff[j] = data.normals[i].dot(eig.eigenvectors().col(j));
				Vector3i adcheck = (angle_diff.array().abs() > 0.95).cast<int>();

				// Check if it needs to be erased
				bool check = (adcheck.array() * diff_bool.array()).any();

				// Discard
				if (!check)
					data.labels[i] = -1;
				else
					// Add to normal matrix
					norm_matrix += (data.normals[i] * data.normals[i].transpose());
			}
		}
	}
}
