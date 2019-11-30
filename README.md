# FASTR
Project Fixed-wing Autonomous System Transition Race (FASTR) involves an autonomous fixed-wing drone racing through a circuit. The work presented in this repository is a step towards building autonomy into drones. An off-the-shelf 'Tello' quadrotor was used for testing utilising its provided SDK commands, computer vision algorithms to detect gates, external localisation using OptiTrack, and many PID controllers for navigation.

See [this](https://www.youtube.com/watch?v=AfEbYtLG2M0) video for a demonstration.

This code was tested on the Ubuntu (18.04.3 LTS) operating system and Python 3.6.8. It required the following Python modules:
- OpenCV 4.0.1 (compiled from source for Python3, with GTK v3, and standard Video I/O (like FFMPEG))
- NumPy
- Scipy
- Matplotlib
- libh264decoder (This is a 3rd party module, compiled from source for Python3. This generated the libh264decoder.so file, if you cannot directly import this then try to compile from source [here](https://github.com/DaWelter/h264decoder/tree/dev). Note that I had to modify a .cpp file to match file given by DJI [here](https://github.com/dji-sdk/Tello-Python/blob/master/Tello_Video/h264decoder/h264decoder.cpp). This amended source code is available in the h264decoder_dev folder.)
- plus other standard modules such as time, logging etc.