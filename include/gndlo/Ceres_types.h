#ifndef CeresTypes_H
#define CeresTypes_H

#include <Eigen/Core>

// Point-to-plane distance but only 2D motion
struct Motion2DCostFunc {
    Motion2DCostFunc(Eigen::Vector3d point, Eigen::Vector3d center,
		Eigen::Vector3d normal, Eigen::Vector3d axis_tx,
		Eigen::Vector3d axis_ty, Eigen::Vector3d axis_rot)
        : point_(point), center_(center), normal_(normal),
		  axis_tx_(axis_tx), axis_ty_(axis_ty), axis_rot_(axis_rot) {}

        template <typename T>
        bool operator()(const T* const rot,
						const T* const d,
                        T* residual) const {

			Eigen::Matrix<T,3,1> pointT = point_.template cast<T>();
			Eigen::Matrix<T,3,1> axis_rotT = axis_rot_.template cast<T>();
			Eigen::Matrix<T,3,1> point_rot = cos(rot[0])*pointT + sin(rot[0])*axis_rotT.cross(pointT)
											 + (1. - cos(rot[0]))*(axis_rotT.dot(pointT))*axis_rotT;
            Eigen::Matrix<T,3,1> difference = center_.template cast<T>() - point_rot
											  - d[0]*axis_tx_.template cast<T>()
											  - d[1]*axis_ty_.template cast<T>();
            residual[0] = difference.dot(normal_.template cast<T>());

            return true;
        }

    private:
		// Axes of rotation and translation
		const Eigen::Vector3d axis_tx_;
		const Eigen::Vector3d axis_ty_;
		const Eigen::Vector3d axis_rot_;
        // Observations for a sample.
        const Eigen::Vector3d point_;
        const Eigen::Vector3d center_;
        const Eigen::Vector3d normal_;
};

// Point-to-plane distance but only 2D motion (inverse tranformation)
struct Motion2DInvCostFunc {
    Motion2DInvCostFunc(Eigen::Vector3d point, Eigen::Vector3d center,
		Eigen::Vector3d normal, Eigen::Vector3d axis_tx,
		Eigen::Vector3d axis_ty, Eigen::Vector3d axis_rot)
        : point_(point), center_(center), normal_(normal),
		  axis_tx_(axis_tx), axis_ty_(axis_ty), axis_rot_(axis_rot) {}

        template <typename T>
        bool operator()(const T* const rot,
						const T* const d,
                        T* residual) const {

			Eigen::Matrix<T,3,1> pointT = point_.template cast<T>()
											- d[0]*axis_tx_.template cast<T>()
											- d[1]*axis_ty_.template cast<T>();
			Eigen::Matrix<T,3,1> axis_rotT = axis_rot_.template cast<T>();
			Eigen::Matrix<T,3,1> point_rot = cos(-rot[0])*pointT + sin(-rot[0])*axis_rotT.cross(pointT)
											 + (1. - cos(-rot[0]))*(axis_rotT.dot(pointT))*axis_rotT;
            Eigen::Matrix<T,3,1> difference = center_.template cast<T>() - point_rot;
            residual[0] = difference.dot(normal_.template cast<T>());

            return true;
        }

    private:
		// Axes of rotation and translation
		const Eigen::Vector3d axis_tx_;
		const Eigen::Vector3d axis_ty_;
		const Eigen::Vector3d axis_rot_;
        // Observations for a sample.
        const Eigen::Vector3d point_;
        const Eigen::Vector3d center_;
        const Eigen::Vector3d normal_;
};

#endif
