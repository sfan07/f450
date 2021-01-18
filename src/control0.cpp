
#include <ros/ros.h>
#include <iostream>
#include <string>
#include <nav_msgs/Odometry.h>
#include <mavros_msgs/PositionTarget.h>

// #include <mavros_msgs/CommandBool.h>
// #include <mavros_msgs/SetMode.h>
// #include <mavros_msgs/State.h>

using namespace std;                    
nav_msgs::Odometry localposition;
void gpslocal_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
    localposition = *msg;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "control0");
    ros::NodeHandle nh("~");

    std::string uavid = "/uav";
    char* droneID = NULL;
    char droneDefaultID = '0';
    if ( argc > 1) {    // if ID is specified as the second argument 
        ROS_INFO("UAV ID specified as: uav%s", argv[1]);
        droneID = argv[1];
    } else {  // if ID is not specified, then set the drone to UAV0
        droneID = &droneDefaultID;
        ROS_WARN("NO UAV ID is specified, set the ID to 0!");
    }
    uavid.push_back(*droneID); // add uav prefixes to topic strings 

    
    ros::Subscriber gpslocal = nh.subscribe<nav_msgs::Odometry>(uavid + "/mavros/global_position/local", 10, gpslocal_callback);

	ros::Publisher setlocal = nh.advertise<mavros_msgs::PositionTarget>(uavid + "/mavros/setpoint_raw/local", 1);
    mavros_msgs::PositionTarget desired_local;

    ros::Rate loop_rate(10.0);
    while(ros::ok()) {
        desired_local.header.stamp = ros::Time::now();
        desired_local.coordinate_frame = 1;
        //Bitmask toindicate which dimensions should be ignored (1 means ignore,0 means not ignore; Bit 10 must set to 0)
        //Bit 1:x, bit 2:y, bit 3:z, bit 4:vx, bit 5:vy, bit 6:vz, bit 7:ax, bit 8:ay, bit 9:az, bit 10:is_force_sp, bit 11:yaw, bit 12:yaw_rate
        //Bit 10 should set to 0, means is not force sp
        desired_local.type_mask = 0b100111111000;
        desired_local.position.x = 1;
        desired_local.position.y = 2;
        desired_local.position.z = 3;
        // if (mask[0]){
        // }
        // else if (mask[1]){
        //     desired_local.type_mask = 0b100111000111;
        //     desired_local.velocity.x = target[0];
        //     desired_local.velocity.y = target[1];
        //     desired_local.velocity.z = target[2];
        // }
        // else if (mask[2]){
        //     desired_local.type_mask = 0b100000111111;
        //     desired_local.acceleration_or_force.x = target[0];
        //     desired_local.acceleration_or_force.y = target[1];
        //     desired_local.acceleration_or_force.z = target[2];
        // }
        desired_local.yaw = 0.0;
        setlocal.publish(desired_local);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;

}