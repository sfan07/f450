gnome-terminal --window -e 'bash -c "gazebo --verbose worlds/iris_arducopter_runway.world"' \
--tab -e 'bash -c "sleep 5; sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map; exec bash"' \
--tab -e 'bash -c "sleep 10; roslaunch f450 f450_apm.launch fcu_url:=udp://127.0.0.1:14551@14555; exec bash"' \
--tab -e 'bash -c "sleep 15; roslaunch px4_command px4_multidrone_pos_estimator_outdoor.launch uavID:=uav1; exec bash"' \
--tab -e 'bash -c "sleep 15; roslaunch px4_command px4_multidrone_pos_controller_outdoor.launch uavID:=uav1; exec bash"' \
--tab -e 'bash -c "sleep 15; rosrun outdoor_gcs outdoor_gcs"' \
# --tab -e 'bash -c "sleep 10; rosrun f450 ros_pathplan.py"' \



