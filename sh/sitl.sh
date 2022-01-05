gnome-terminal --window -e 'bash -c "roslaunch f450 multi_uav_mavros_sitl_4.launch"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav1; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav1; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav2; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav2; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav3; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav3; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav4; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav4; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav5; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav5; exec bash"' \
--tab -e 'bash -c "sleep 10; rosrun outdoor_gcs outdoor_gcs"' \
--tab -e 'bash -c "sleep 10; rosrun f450 ros_pso_pathplan.py"' \


# gnome-terminal --window -e 'bash -c "roslaunch f450 multi_uav_mavros_sitl_1.launch"' \
# gnome-terminal --window -e 'bash -c "roslaunch f450 multi_uav_mavros_sitl_1_one_obstacle.launch"' \
# gnome-terminal --window -e 'bash -c "roslaunch f450 multi_uav_mavros_sitl_1_three_obstacle.launch"' \
# gnome-terminal --window -e 'bash -c "roslaunch f450 multi_uav_mavros_sitl_4_one_obstacle.launch"' \
# gnome-terminal --window -e 'bash -c "roslaunch f450 multi_uav_mavros_sitl_4_three_obstacle.launch"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav6; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav6; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav7; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav7; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav8; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav8; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav9; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav9; exec bash"' \

