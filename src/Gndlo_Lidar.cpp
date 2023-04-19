#include "Gndlo_Lidar.h"

#include <math.h>

using namespace Eigen;


void GNDLO_Lidar::initializeSensor()
{
    // Report
	if (options.flag_verbose)
	{
		std::cout << "Lidar sensor parameters: " << std::endl;
		std::cout << "\tResolution: " << sensor.cols << ", " << sensor.rows << std::endl;
		std::cout << "\tFlag linear vertical angles: " << sensor.flag_linear_vert << std::endl;
		std::cout << "\tHorizontal range: " << sensor.hor_min << " to " << sensor.hor_max << std::endl;
		if (sensor.flag_linear_vert)
			std::cout << "\tVertical range: " << sensor.ver_min << " to " << sensor.ver_max << std::endl;
		else
			std::cout << "\tVertical range: " << sensor.ver_angles[0] << " to " << sensor.ver_angles[rows-1] << std::endl;
	}

	// Set the resolution
	rows = sensor.rows;
	cols = sensor.cols;

    // Define vertical angles
	if (sensor.flag_linear_vert)
		for (int i = 0; i < rows; ++i)
	    {
			float ang = sensor.ver_min + (sensor.ver_max - sensor.ver_min) * (float(i)/float(rows-1));
	        theta.push_back(ang);
		}
	else
	{
		theta = sensor.ver_angles;
	}

    // Create vectors of horizontal angles
    for (int i = 0; i < cols; ++i)
    {
        float ang = sensor.hor_min + (sensor.hor_max - sensor.hor_min) * (float(i)/float(cols-1));
        phi.push_back(ang);
    }

    // Populate sin and cos
		// Theta
    for (int i = 0; i < theta.size(); ++i)
    {
        sint.push_back(sin(theta[i]));
        cost.push_back(cos(theta[i]));
    }
		// Phi
    for (int i = 0; i < phi.size(); ++i)
    {
        sinp.push_back(sin(phi[i]));
        cosp.push_back(cos(phi[i]));
    }

}

void GNDLO_Lidar::calculateXYZ(Level & lvl)
{
    unsigned int rr, cc;
    lvl.getResolution(rr, cc);

    lvl.x.resize(rr,cc);
    lvl.y.resize(rr,cc);
    lvl.z.resize(rr,cc);

    for (unsigned int u=0; u<cc; ++u)
        for (unsigned int v=0; v<rr; ++v)
        {
            lvl.x(v,u) = lvl.d(v,u) * cost[v] * cosp[u];
            lvl.y(v,u) = lvl.d(v,u) * cost[v] * sinp[u];
            lvl.z(v,u) = lvl.d(v,u) * sint[v];
        }
}

void GNDLO_Lidar::transformPixels(const std::vector<Eigen::Vector3f> & centers,
                                    const Eigen::Matrix4f & transform,
                                    std::vector<Eigen::Vector2i> & outpixels)
{
    // Clear output vector
    outpixels.clear();

    Vector3f point;
    for (int i=0; i<centers.size(); ++i)
	{
        // Obtain the transformed point
		point = transform.block<3,3>(0,0)*centers[i] + transform.block<3,1>(0,3);
        float x = point[0], y = point[1], z = point[2];

        // Transform to spherical coordinates
        float r = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
        float p = atan2(y,x);
        float t = asin(z/r);

        // Get index of the horizontal angle
		float phi_idxf = (p - sensor.hor_min)*(cols-1)/(sensor.hor_max - sensor.hor_min);
        int phi_idx = round(phi_idxf);

        // Get index of the vertical angle
		int theta_idx;
		if (sensor.flag_linear_vert)
		{
			float theta_idxf = (t - sensor.ver_min)*(rows-1)/(sensor.ver_max - sensor.ver_min);
	        theta_idx = round(theta_idxf);
		}
		else
		{
			auto theta_it = lower_bound(theta.begin(), theta.end(), t, std::greater<float>());
			theta_idx = distance(theta.begin(), theta_it);
			if ( abs(theta[theta_idx+1] - t) < abs(t - theta[theta_idx]) )
			++theta_idx;
		}

        // Save the new pixel location
		Vector2f auxpix;
		auxpix(0) = theta_idx;
		auxpix(1) = phi_idx;
		outpixels.push_back(auxpix.array().round().cast<int>());
	}
}
