gnome-terminal --window -e 'bash -c "roscore"' \
--tab -e 'bash -c "sleep 2; rosrun outdoor_gcs outdoor_gcs"' \
--tab -e 'bash -c "sleep 2; rosrun f450 local_launch.sh ros1 odroid@192.168.2.179 uav1; exec bash"' \
# gnome-terminal --window -e 'bash -c "roslaunch f450 f450_test.launch uavID:=uav1 tgt_system:=1; exec bash"' \
# --tab -e 'bash -c "sleep 2; roslaunch f450 f450_test.launch uavID:=uav2 tgt_system:=2; exec bash"' \
# --tab -e 'bash -c "sleep 2; rosrun outdoor_gcs outdoor_gcs; exec bash"' \
# --tab -e 'bash -c "sleep 2; roslaunch f450 f450_test.launch uavID:=uav3 tgt_system:=3; exec bash"' \
# --tab -e 'bash -c "sleep 2; roslaunch f450 f450_test.launch uavID:=uav4 tgt_system:=4; exec bash"' \
# --tab -e 'bash -c "sleep 2; roslaunch f450 f450_test.launch uavID:=uav5 tgt_system:=5; exec bash"' \


