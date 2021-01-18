# gnome-terminal --window -e 'bash -c "rosservice call /uav1/mavros/cmd/arming "value: true"; exec bash"' \
# --tab -e 'bash -c "sleep 2; rosservice call /uav1/mavros/cmd/arming "value: true"; exec bash"' \
# --tab -e 'bash -c "sleep 2; rosrun outdoor_gcs outdoor_gcs; exec bash"' \