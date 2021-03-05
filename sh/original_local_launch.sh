#!/bin/bash

LEADER=$false

SESSION=$1
HOST_IP=$2
UAV=$3
LEADER=$4

tmuxstart() {
    if [[ $(tmux has-session -t "$1") -eq 0 ]] ; then
        echo "Killing previous session with name $1"
        tmux kill-session -t  "$1"
    fi
    #rest of tmux script to create session named "sess"
    tmux new-session -d -s "$1"
}

splitandrun() {
    tmux send-keys -t $1 "tmux split-window -h $2 && tmux select-layout even-horizontal" ENTER
}

sendcmd() {
    tmux send-keys -t $1 "$2" ENTER
}
rostopic list >/dev/null 2>&1
if [ $? -ne 0 ] ; then
    echo "rosmaster not started! Exiting."
    exit 1
fi

IP=$(sed 's&.*@\(\)&\1&' <<< ${HOST_IP})

until ping -c1 ${IP} >/dev/null 2>&1; do 
    echo "Pinging $IP...";
done

read -p "Destination $IP reached. Press enter to begin tmux session and ssh to remote vehicle"

tmuxstart ${SESSION}

# Split panes then ssh to the vehicle in each pane
splitandrun ${SESSION} "ssh -X ${HOST_IP}"
splitandrun ${SESSION} "ssh -X ${HOST_IP}"
if [ $LEADER ] ; then
    splitandrun ${SESSION} "ssh -X ${HOST_IP}"
fi
# ssh to the vehicle in the original pane
sendcmd 0 "ssh -tt -X ${HOST_IP}"

# Must wait, otherwise panes other than 0 may not initialize properly.
    echo "Wait for panes to fully initialize."

for COUNTDOWN in 3 2 1
do
    echo $COUNTDOWN
    sleep 1
done

sendcmd 0 "roslaunch px4_command mavros_multi_drone.launch uavID:=$UAV"
sendcmd 1 "roslaunch px4_command px4_multidrone_pos_estimator_pure_vision.launch uavID:=$UAV"
sendcmd 2 "roslaunch px4_command px4_multidrone_pos_controller.launch uavID:=$UAV"
if [ $LEADER ] ; then
    sendcmd 3 "roslaunch px4_command px4_interdrone_communication.launch uavID:=$UAV"
fi
## Create the windows on which each node or .launch file is going to run

gnome-terminal --tab -- tmux attach -t ${SESSION}

