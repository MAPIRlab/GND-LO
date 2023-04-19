#include "Gndlo.h"
#include "Gndlo_Lidar.h"

// ROS
#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>


using namespace Eigen;
using namespace std;


// ------------------------------------------------------
//						CLASS
// ------------------------------------------------------
class GNDLO_Node : public rclcpp::Node, public GNDLO_Lidar
{
  public:
	// Constructor
    GNDLO_Node()
    : Node("gndlo_node")
	{
		// Initialize variables
		odom_pose = Matrix4f::Identity();
		odom_cov = MatrixXf::Zero(6,6);

		// Declare parameters
		this->declare_all_parameters();

		// Save parameters in options
		this->get_all_parameters();

		// Create publisher
		pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("gndlo/odom_pose_cov", 10);
		odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("gndlo/odom", 10);

		// Create subscription to image and info
		topic = this->get_parameter("subs_topic").get_parameter_value().get<string>();
		image_sub_.subscribe(this, topic + "/range/image");
    	info_sub_.subscribe(this, topic + "/range/sensor_info");

		// Synchronize subscribers
		sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::LaserScan>>(image_sub_, info_sub_, 20);
    	sync_->registerCallback(std::bind(&GNDLO_Node::image_callback, this, placeholders::_1, placeholders::_2));

		// Open file to save results
		if (options.flag_save_results)
		{
			if (options.results_file_name.size() < 5)
			{
				RCLCPP_WARN(this->get_logger(), "Invalid name for results file: ");
				RCLCPP_WARN(this->get_logger(), options.results_file_name.c_str());
				options.flag_save_results = false;
			}
			else
			{
				if (options.flag_verbose)
					RCLCPP_INFO(this->get_logger(), "Saving results to ");
					RCLCPP_INFO(this->get_logger(), options.results_file_name.c_str());
				results_file.open(options.results_file_name);
			}
		}
    }

	// Destructor
	~GNDLO_Node()
	{
		// Ensure results file is closed
		if (results_file.is_open())
			results_file.close();
	}

  private:
	//------------------------------------
	// PARAMETERS
	//------------------------------------
  	// Declare parameters
	void declare_all_parameters()
	{
		// Declare parameters
			// ROS topic to subscribe
		this->declare_parameter("subs_topic", "/kitti");
			// General
		this->declare_parameter("num_threads", 8);
		this->declare_parameter("valid_ratio", 0.8);
			// Flags
		this->declare_parameter("flag_verbose", true);
		this->declare_parameter("flag_flat_blur", true);
		this->declare_parameter("flag_solve_backforth", true);
		this->declare_parameter("flag_filter", true);
			// Gaussian filtering
		this->declare_parameter("select_radius", 1);
		this->declare_parameter("gaussian_sigma", 0.5);
			// Quadtree selection
		this->declare_parameter("quadtrees_avg", 0.1);
		this->declare_parameter("quadtrees_std", 0.015);
		this->declare_parameter("quadtrees_min_lvl", 2);
		this->declare_parameter("quadtrees_max_lvl", 5);
			// Orthog Culling
		this->declare_parameter("count_goal", 50.);
		this->declare_parameter("starting_size", 4);
			// Ground clustering
		this->declare_parameter("ground_threshold_deg", 10.);
		this->declare_parameter("wall_threshold_deg", 60.);
			// Solution
		this->declare_parameter("iterations", 5);
		this->declare_parameter("huber_loss", 3e-5);
		this->declare_parameter("trans_bound", 1.);
			// Convergence
		this->declare_parameter("pix_threshold", 5.);
		this->declare_parameter("trans_threshold", 0.002);
		this->declare_parameter("rot_threshold", 0.5*(M_PI/180.));
			// Filter
		this->declare_parameter("filter_kd", 100.);
		this->declare_parameter("filter_pd", 0.);
		this->declare_parameter("filter_kf", 2.);
		this->declare_parameter("filter_pf", 1.);
			// Output
		this->declare_parameter("flag_save_results", true);
		this->declare_parameter("results_file_name", "");
	}

	// Get parameters
	void get_all_parameters()
	{
		// Set options
			// General
		options.num_threads = this->get_parameter("num_threads").get_parameter_value().get<int>();
		options.valid_ratio = this->get_parameter("valid_ratio").get_parameter_value().get<double>();
			// Flags
		options.flag_verbose = this->get_parameter("flag_verbose").get_parameter_value().get<bool>();
		options.flag_flat_blur = this->get_parameter("flag_flat_blur").get_parameter_value().get<bool>();
		options.flag_solve_backforth = this->get_parameter("flag_solve_backforth").get_parameter_value().get<bool>();
		options.flag_filter = this->get_parameter("flag_filter").get_parameter_value().get<bool>();
			// Gaussian filtering
		options.select_radius = this->get_parameter("select_radius").get_parameter_value().get<int>();
		options.gaussian_sigma = this->get_parameter("gaussian_sigma").get_parameter_value().get<double>();
			// Quadtree selection
		options.quadtrees_avg = this->get_parameter("quadtrees_avg").get_parameter_value().get<double>();
		options.quadtrees_std = this->get_parameter("quadtrees_std").get_parameter_value().get<double>();
		options.quadtrees_min_lvl = this->get_parameter("quadtrees_min_lvl").get_parameter_value().get<int>();
		options.quadtrees_max_lvl = this->get_parameter("quadtrees_max_lvl").get_parameter_value().get<int>();
			// Orthog Culling
		options.count_goal = this->get_parameter("count_goal").get_parameter_value().get<double>();
		options.starting_size = this->get_parameter("starting_size").get_parameter_value().get<int>();
			// Ground clustering
		options.ground_threshold_deg = this->get_parameter("ground_threshold_deg").get_parameter_value().get<double>();
		options.wall_threshold_deg = this->get_parameter("wall_threshold_deg").get_parameter_value().get<double>();
			// Solution
		options.iterations = this->get_parameter("iterations").get_parameter_value().get<int>();
		options.huber_loss = this->get_parameter("huber_loss").get_parameter_value().get<double>();
		options.trans_bound = this->get_parameter("trans_bound").get_parameter_value().get<double>();
			// Convergence
		options.pix_threshold = this->get_parameter("pix_threshold").get_parameter_value().get<double>();
		options.trans_threshold = this->get_parameter("trans_threshold").get_parameter_value().get<double>();
		options.rot_threshold = this->get_parameter("rot_threshold").get_parameter_value().get<double>();
			// Filter
		options.filter_kd = this->get_parameter("filter_kd").get_parameter_value().get<double>();
		options.filter_pd = this->get_parameter("filter_pd").get_parameter_value().get<double>();
		options.filter_kf = this->get_parameter("filter_kf").get_parameter_value().get<double>();
		options.filter_pf = this->get_parameter("filter_pf").get_parameter_value().get<double>();
			// Output
		options.flag_save_results = this->get_parameter("flag_save_results").get_parameter_value().get<bool>();
		options.results_file_name = this->get_parameter("results_file_name").get_parameter_value().get<string>();
	}


	//------------------------------------
	// PUBLISHER
	//------------------------------------
	// Publish Pose with Covariance
	void publish_odometry(const std_msgs::msg::Header & header, const Eigen::Matrix4f & pose, const Eigen::MatrixXf & cov)
	{
		// Create message PoseWithCovarianceStamped
		auto message = geometry_msgs::msg::PoseWithCovarianceStamped();
			// Set header
		message.header = header;
		message.header.frame_id = "world";
			// Set pose
		Eigen::Quaternionf quat(pose.block<3,3>(0,0));
		message.pose.pose.position.x = pose(0,3);
		message.pose.pose.position.y = pose(1,3);
		message.pose.pose.position.z = pose(2,3);
		message.pose.pose.orientation.x = quat.x();
		message.pose.pose.orientation.y = quat.y();
		message.pose.pose.orientation.z = quat.z();
		message.pose.pose.orientation.w = quat.w();
			// Set covariance
		Map<Matrix<double, 6, 6, RowMajor>>(begin(message.pose.covariance)) = cov.cast<double>();

		// Create odometry message
		auto odom_msg = nav_msgs::msg::Odometry();
			// Set header
		odom_msg.header = header;
		odom_msg.header.frame_id = "world";
		odom_msg.child_frame_id = "lidar";
			// Set pose as above
		odom_msg.pose = message.pose;

		// Publish
		pose_pub_->publish(message);
		odom_pub_->publish(odom_msg);
	}


	//------------------------------------
	// IMAGE+INFO CALLBACK
	//------------------------------------
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg, const sensor_msgs::msg::LaserScan::ConstSharedPtr& info_msg)
    {
		if (options.flag_verbose)
			RCLCPP_INFO(this->get_logger(), "Pair of messages (image - info) received:");

		// Save header
		header = img_msg->header;

		// Save image
        cv_bridge::CvImagePtr cv_img = cv_bridge::toCvCopy(img_msg);
        cv::cv2eigen(cv_img->image, input_img);

		// Save Sensor information
		sensor.flag_linear_vert = true;
		sensor.rows = info_msg->range_min;
		sensor.cols = info_msg->range_max;
	    sensor.hor_min = info_msg->angle_min;
		sensor.hor_max = info_msg->angle_max;
		sensor.ver_min = info_msg->ranges[0];
		sensor.ver_max = info_msg->ranges[sensor.rows-1];
		for (int i=0; i<sensor.rows; ++i)
			sensor.ver_angles.push_back(info_msg->ranges[i]);

		// Initialize when first data arrives
		if (first_data)
		{
			// Initialize
			first_data = false;
			this->setSensorParameters(sensor);
			this->setDepthInput(input_img);
			this->initialize();

			// Create header in results file
			if (options.flag_save_results)
				results_file << "#time \ttx \tty \ttz \tqx \tqy \tqz \tqw\n";

		}
		// Do the normal calculations
		else
		{
			// Set input frame
			this->setDepthInput(input_img);

			// Odometry
			this->runOdometry();

			// Display execution time
			if (options.flag_verbose)
			{
				RCLCPP_INFO(this->get_logger(), "Frame: %li", this->number_frames);
				RCLCPP_INFO(this->get_logger(), "Execution time: %f ms", this->execution_time);
				RCLCPP_INFO(this->get_logger(), "Average execution time: %f ms:", this->avg_exec_time);
				RCLCPP_INFO(this->get_logger(), "\t-Flatness: %f ms", this->avg_time_bd(0));
				RCLCPP_INFO(this->get_logger(), "\t-Quadtree: %f ms", this->avg_time_bd(1));
				RCLCPP_INFO(this->get_logger(), "\t-Plane fitting: %f ms", this->avg_time_bd(2));
				RCLCPP_INFO(this->get_logger(), "\t-Labeling: %f ms", this->avg_time_bd(3));
				RCLCPP_INFO(this->get_logger(), "\t-Culling: %f ms", this->avg_time_bd(4));
				RCLCPP_INFO(this->get_logger(), "\t-Ground clustering: %f ms", this->avg_time_bd(5));
				RCLCPP_INFO(this->get_logger(), "\t-Ground alignment: %f ms", this->avg_time_bd(6));
				RCLCPP_INFO(this->get_logger(), "\t-Point matching: %f ms", this->avg_time_bd(7));
				RCLCPP_INFO(this->get_logger(), "\t-Motion estimation: %f ms", this->avg_time_bd(8));
				RCLCPP_INFO(this->get_logger(), "\t-Iterations: %f ms", this->avg_time_bd(9));
				RCLCPP_INFO(this->get_logger(), "\t-Motion filter: %f ms\n", this->avg_time_bd(10));
			}
		}

		// Get results
		this->getPose(odom_pose);
		this->getCovariance(odom_cov);

		// Publish the results
		publish_odometry(header, odom_pose, odom_cov);

		// Save results to file
		if (options.flag_save_results)
		{
			Quaternionf q(odom_pose.block<3,3>(0,0));
			rclcpp::Time timestamp(header.stamp);
			char timestr[20];
			snprintf(timestr, sizeof(timestr), "%.9f", timestamp.seconds());
			results_file << timestr << " "
			 			 << odom_pose(0,3) << " " << odom_pose(1,3) << " " << odom_pose(2,3) << " "
			 			 << q.vec()(0) << " " << q.vec()(1) << " " << q.vec()(2) << " " << q.w()
						 << endl;
		}
    }


	//------------------------------------
	// VARIABLES
	//------------------------------------
	// Variables
	string topic = "/kitti";
	bool first_data = true;
	Eigen::MatrixXf input_img;
	Eigen::Matrix4f odom_pose;
	Eigen::MatrixXf odom_cov;
	GNDLO_Lidar::SensorInfo sensor;

	// Output results file
	ofstream results_file;

	// Declare publishers
	std_msgs::msg::Header header;
	rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
	rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

	// Declare subscriptions and synchronizer
	message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
	message_filters::Subscriber<sensor_msgs::msg::LaserScan> info_sub_;
	std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::LaserScan>> sync_;
};


// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------


int main(int num_arg, char *argv[])
{
	//						Start
	//----------------------------------------------------------------------
	try
	{
		// Start ROS
		//----------------------------------------------------------------------
		cout << "GNDLO Odometry node: READY." << endl;
	    rclcpp::init(num_arg, argv);
		rclcpp::spin(std::make_shared<GNDLO_Node>());
		rclcpp::shutdown();

		cout << "GNDLO Odometry node: SHUTTING DOWN" << endl;

		return 0;

	}
	catch (std::exception &e)
	{
		cout << "Exception caught: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		printf("Untyped exception!!");
		return -1;
	}
}
