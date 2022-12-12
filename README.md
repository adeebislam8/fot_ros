# fot_ros
Package to implement the frenet optimal trajectory local planner using ROS

run the launch file to launch the package
or 
rosrun the "run_fot.py" script

fot.py contains functions needed to calculate the paths 

At first i tried to implement the entrie thing in python. Since that implementaion was slow I tried to code some of the functions, such as finding the nearest points, calculating the frenet coordinate etc in cpp, and then use c_types to speed up the implementation. The speed up unfortunately was not as significant. These are written in fot_cpp_1.cpp

ignore run_fot_standalone.py
