## ROS Drone Project
The goal of this project is that given a heterogeneous, ROS-simulated environment to identify and read clue-boards scattered throughout the course.

The solution that we came up with was a package containing two robots: a control drone and the execution drone. The control drone behaves as an overhead observer making global observations
about the environment. It provides the execution drone with a set of global coordinates and streams position data of possible targets. The execution drone follows the observers commands, exploring,
identifying and reading clue-boards when they are reached. 

For more information on the implementation details of the controllers used to stabilize the drones in a windy environment, the CNN architecture and computer vision tools used to identify and read the clue-boards, 
and ROS implementation details please see the `report.pdf`.
