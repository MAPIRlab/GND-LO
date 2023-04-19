
#ifndef Lidar_H
#define Lidar_H

#include "Gndlo.h"


class  GNDLO_Lidar : public GNDLO {
public:
    // Sensor parameters struct
    // -----------------------------------------
    struct SensorInfo {
        int rows;						// Image resolution: rows
        int cols;						// Image resolution: columns
        float hor_min = M_PI;			// Horizontal min angle
        float hor_max = -M_PI;			// Horizontal max angle
		float ver_min;					// Vertical min angle
		float ver_max;					// Vertical max angle
        std::vector<float> ver_angles;	// Vector of vertical angles (if not linearly distributed)
		bool flag_linear_vert = 0;		// Flag: vertical angles are linearly distributed
    };

protected:

    // Angles vectors: input, pyramid, and pyramid of cos and sin
    // -----------------------------------------
    SensorInfo sensor;							// Sensor information (defined above)
    std::vector<float> theta, phi;				// Angle vectors
    std::vector<float> sint, cost, sinp, cosp;	// Cosine and sine of angle vectors

    // Basic sensor functions
    // -----------------------------------------
    void initializeSensor();

    void calculateXYZ(Level & lvl);

	void transformPixels(const std::vector<Eigen::Vector3f> & centers,
                            const Eigen::Matrix4f & transform,
                            std::vector<Eigen::Vector2i> & outpixels);

public:
    // Constructor
    GNDLO_Lidar(){}

    inline void setSensorParameters(SensorInfo in) {sensor = in;}

    /** Set vertical resolution vector */
    inline void setVRes(std::vector<float> vert_res) {theta = vert_res;}

    /** Set horizontal resolution vector */
    inline void setHRes(std::vector<float> hor_res) {phi = hor_res;}

};


#endif
