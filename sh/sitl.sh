gnome-terminal --window -e 'bash -c "roslaunch px4 multi_uav_mavros_sitl.launch"' \
--tab -e 'bash -c "sleep 20; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav1; exec bash"' \
--tab -e 'bash -c "sleep 20; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav1; exec bash"' \
--tab -e 'bash -c "sleep 2; rosrun outdoor_gcs outdoor_gcs"' \



