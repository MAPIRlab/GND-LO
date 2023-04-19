#include "Gndlo.h"

using namespace Eigen;
using namespace std;
using namespace cv;

void GNDLO::obtainPlanarityCurvature(const Level & lvl, int ksize, MatrixXf & out)
{
	// Set variables
	unsigned int rr, cc;
	lvl.getResolution(rr, cc);
	Matrix<float, Dynamic, Dynamic, RowMajor> in = lvl.d.cwiseInverse();

	// Set OpenCV Mat
	cv::Mat cvim(rr, cc, CV_32FC1, in.data());
	cv::Mat dhim, dvim;

	// Erase infinites
	for (int col=0; col<cc; ++col)
		for (int row=0; row<rr; ++row)
			if (lvl.d(row,col) == 0.)
				in(row,col) = 0.;

	// Apply Sobel
	cv::Sobel(cvim, dhim, -1, 2, 0, ksize, 1, 0, BORDER_ISOLATED);
	cv::Sobel(cvim, dvim, -1, 0, 2, ksize, 1, 0, BORDER_ISOLATED);

	// Read in eigen
	Map<Matrix<float, Dynamic, Dynamic, RowMajor>> dh(dhim.ptr<float>(), rr, cc);
	Map<Matrix<float, Dynamic, Dynamic, RowMajor>> dv(dvim.ptr<float>(), rr, cc);

	// Sum to get the flatness
	out = (dh.cwiseAbs() + dv.cwiseAbs()).cwiseSqrt();
}
