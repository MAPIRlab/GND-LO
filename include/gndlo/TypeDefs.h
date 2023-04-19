#ifndef TypeDefs_H
#define TypeDefs_H

#include <Eigen/Core>
#include <vector>

// Bool matrix
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

// Structure for twist with associated covariance
class TwistCov
{
public:
	Eigen::Matrix<float, 6, 1> twist = Eigen::Matrix<float, 6, 1>::Zero();
	Eigen::Matrix<float, 6, 6> cov = Eigen::Matrix<float, 6, 6>::Zero();

	TwistCov& operator= (TwistCov other)
	{
		this->twist = other.twist;
		this->cov = other.cov;

		return *this;
	}

	inline void setZero()
	{
		twist.setZero();
		cov.setZero();
	}

	TwistCov(){	this->setZero(); }
};

// Structure for derivatives in u, v and t
class DerivativeMat
{
public:
    Eigen::MatrixXf u;
    Eigen::MatrixXf v;
    Eigen::MatrixXf t;
};

// Structure for ground
class Ground
{
public:
	Eigen::Vector3f center = Eigen::Vector3f::Zero();		// Center of the ground plane
	Eigen::Vector3f normal = Eigen::Vector3f::Zero();		// Normal of the ground plane
	int count_patches = 0;									// Count of participating patches
	int count_px = 0;										// Count of participating pixels
	Eigen::Matrix<float, 6, 6> covariance = Eigen::Matrix<float, 6, 6>::Zero(); // Covariance of [c,n] (6x6)

	Ground& operator= (Ground other)
	{
		this->clear();
		this->center = other.center;
		this->normal = other.normal;
		this->count_patches = other.count_patches;
		this->count_px = other.count_px;
		this->covariance = other.covariance;

		return *this;
	}

	inline void clear()
	{
		center.setZero();
		normal.setZero();
		count_patches = 0;
		count_px = 0;
		covariance = Eigen::MatrixXf::Zero(6,6);
	}
};

// Structure for the set of features obtained using quadtrees
class SizedData
{
public:
	std::vector<Eigen::Matrix3f> covars;	// Covariance matrices on Z0
	std::vector<Eigen::Vector3f> normals;   // Normal vectors on Z0
    std::vector<Eigen::Vector3f> centers;   // Center of planes on Z0
    std::vector<float> fitnesses;           // Fitness of planes on Z0
    std::vector<Eigen::Vector3f> points;    // Points on Z1
    std::vector<Eigen::Vector2i> px0;       // Pixel coords on Z0
    std::vector<Eigen::Vector2i> px1;       // Pixel coords on Z1
	std::vector<int> sizes;					// Size of the patch on the image (side of the square)
	std::vector<int> labels;				// Label: unlabeled (-1), ground (0), walls (1)

	SizedData& operator= (SizedData other)
	{
		this->clear();
		this->covars = other.covars;
		this->normals = other.normals;
		this->centers = other.centers;
		this->fitnesses = other.fitnesses;
		this->points = other.points;
		this->px0 = other.px0;
		this->px1 = other.px1;
		this->sizes = other.sizes;
		this->labels = other.labels;

		return *this;
	}

    inline void clear()
    {
		covars.clear();
        normals.clear();
        centers.clear();
        fitnesses.clear();
        points.clear();
        px0.clear();
        px1.clear();
		sizes.clear();
		labels.clear();
    };

    inline void erase(int i)
    {
		covars.erase(covars.begin() + i);
        normals.erase(normals.begin() + i);
        centers.erase(centers.begin() + i);
        fitnesses.erase(fitnesses.begin() + i);
        points.erase(points.begin() + i);
        px0.erase(px0.begin() + i);
        px1.erase(px1.begin() + i);
		sizes.erase(sizes.begin() + i);
		labels.erase(labels.begin() + i);
    }
};

// Image information
class Level
{

private:
	// Variables
	unsigned int rows, cols;

public:
	// Matrices
	Eigen::MatrixXf i;
	Eigen::MatrixXf d;
	Eigen::MatrixXf x;
	Eigen::MatrixXf y;
	Eigen::MatrixXf z;
	Eigen::MatrixXf planar;
	Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valids;

	// Funcions
	Level& operator= (Level other)
	{
		unsigned int rows_aux, cols_aux;
		other.getResolution(rows_aux, cols_aux);

		this->setResolution(rows_aux, cols_aux);

		this->i = other.i;
		this->d = other.d;
		this->x = other.x;
		this->y = other.y;
		this->z = other.z;
		this->planar = other.planar;
		this->valids = other.valids;

		return *this;
	}

	// Input
	void setResolution(unsigned int r, unsigned int c)
	{
		rows = r;
		cols = c;

		// Resize
		i.resize(rows, cols);
		d.resize(rows, cols);
		x.resize(rows, cols);
		y.resize(rows, cols);
		z.resize(rows, cols);
		planar.resize(rows, cols);
		valids.resize(rows, cols);

		// Set zero
		i.setZero();
		d.setZero();
		x.setZero();
		y.setZero();
		z.setZero();
		planar.setZero();
		valids.setZero();
	}

	// Output
	void getResolution(unsigned int & outrows, unsigned int & outcols) const {outrows = rows; outcols = cols;}
	void getResolution(int & outrows, int & outcols) const {outrows = rows; outcols = cols;}
};

// For pixel sorting
struct pxComp
{
    constexpr bool operator()(
        std::pair<float, Eigen::Vector2i> const& a,
        std::pair<float, Eigen::Vector2i> const& b)
        const noexcept
    {
        return a.first > b.first;
    }
};

#endif
