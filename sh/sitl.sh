gnome-terminal --window -e 'bash -c "roslaunch f450 multi_uav_mavros_sitl.launch"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav1; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav2; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav1; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav2; exec bash"' \
--tab -e 'bash -c "sleep 10; rosrun outdoor_gcs outdoor_gcs"' \
--tab -e 'bash -c "sleep 10; rosrun f450 ros_f_pathplan.py"' \


# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav3; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav4; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav5; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav3; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav4; exec bash"' \
# --tab -e 'bash -c "sleep 10; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav5; exec bash"' \

