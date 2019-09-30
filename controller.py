import numpy as np
import config
from tello_methods import Tello
from localisation import trueState

def runWaypoint(streamingClient, r_wd, dt, tello):
	'''
	Controller for waypoint navigation using OptiTrack (streamingClient) and 
	given vector to next waypoint (r_wd). Projects the waypoint vector into the
	x-y plane and calculates the yaw offset and the forward/back and left/right
	combinations of inputs required to directly track the waypoint.
	'''
	true_state = trueState(streamingClient)
	current_orient_euler = true_state[3]

	# Vector projection along x-y plane (height component (z) is zero)
	r_wd_proj = np.array([r_wd[0], r_wd[1], 0])

	# Find the yaw angle (between the vector and the x-axis) through the dot
	# product formula. This gives the yaw angle required to face the waypoint.
	yaw_w = np.arctan2(r_wd_proj[1], r_wd_proj[0])		# in radians

	# Offset angle between drone heading and waypoint heading angle (yaw)
	yaw_w = yaw_w * 180/np.pi
	beta = yaw_w - (current_orient_euler[2] * 180/np.pi)
	# Update frequency is too slow with the Tello's onboard state readings
	# state = tello.readState()
	# beta = yaw_w - state['yaw']		# in degree

	# yaw_w can give values from -180 to 180, the Tello state can range from
	# -180 to 180 hence max beta: 360, min beta = -360.
	# Correct the angle to shortest rotation if exceeding 180 degrees
	if beta > 180:
		beta = beta - 360
	elif beta < -180:
		beta = beta + 360

	# Use the angle to find components of the vector projection in the forward/
	# back direction and the left/right direction.
	signal = np.array([np.linalg.norm(r_wd_proj) * np.sin(beta * np.pi/180),	# Lateral
		np.linalg.norm(r_wd_proj) * np.cos(beta * np.pi/180),	# Longitudinal
		r_wd[2],			# Vertical
		beta])				# yaw

	reference = np.array([0, 0, 0, 0])
	error = signal - reference

	try:
		controllerWaypoint(error, runWaypoint.prev_error, dt, tello)
	except AttributeError:
		controllerWaypoint(error, error, dt, tello)		# first run

	runWaypoint.prev_error = error
	return error


def controllerWaypoint(error, prev_error, dt, tello):
	''' PD controller for navigating to waypoints. '''

	# Numerical differentiation - first order difference scheme
	error_dot = (error - prev_error) / dt

	# PD constants and controller (Standard form)
	Kp = np.array([1, 0.4, 1, 10])	# lr, fb, ud, yaw
	Td = np.array([0, 0, 0, 0])
	pid_input = Kp * (error + Td * error_dot)

	# Longitudinal to laterial ratio
	ratio = pid_input[1] / pid_input[0]

	# Maintain ratio between the limited controller inputs
	pid_input = controllerLimits(pid_input, -100.0, 100.0)
	if abs(ratio) > 1:
		pid_input[0] = (1 / ratio) * pid_input[1]
		#component[1] = limited_component[1]
	else:
		pid_input[1] = ratio * pid_input[0]
		#component[0] = limited_component[0]
	# tello.rc(yaw=int(pid_input[3]))
	tello.rc(lr=int(pid_input[0]), fb=int(pid_input[1]), ud=-int(pid_input[2]), 
			 yaw=int(pid_input[3]))


def runEdgeCont(w, hori_offset, num_of_edges, dist_to_gate, dt, tello):
	'''
	This controller should only run when only 1 edge of the gate is
	detected hence is separate to the PID controller used with fully detected
	gate in runPID.
	'''
	# If both edges are detected
	if num_of_edges == 2:

		# Calculates actual horizontal distance (cm) offset from centre of gate
		dist_offset = config.GATE_WIDTH / w * hori_offset

		# Horizontal angle to centre of gate (rads)
		offset_angle = np.arctan(dist_offset / dist_to_gate)

		# error for horizontal offset
		reference = 0
		signal = offset_angle * 180/np.pi
		error = signal - reference

	# If 1 edge detected meaning drone is close to the gate
	elif num_of_edges == 1:

		# The distance cannot be determined hence magnitude of offset does
		# not indicate the magnitude of the response required therefore a
		# constant error is set to force the drone to move the other way.
		if hori_offset <= 0:
			error = -3
		else:
			error = 3

	# No edges found so just move forward
	else:
		tello.rc(fb=40)
		return
	
	try:
		edgesPD(error, runEdgeCont.prev_error, dt, tello)
	except AttributeError:
		edgesPD(error, error, dt, tello)

	# prev_error needs to be persistent hence is an attribute
	runEdgeCont.prev_error = error
	return error


def edgesPD(error, prev_error, dt, tello):
	'''
	An integral term is negligible since the drone will be close to the gate
	for short periods of time
	'''
 	# Numerical differentiation - first order difference scheme
	error_dot = (error - prev_error) / dt

	# PD constants and controller
	Kp = 3
	Td = 1
	pid_input = -Kp * (error + Td * error_dot)

	# Limit PID input to max of 100, and minimum of -100
	pid_input = controllerLimits(pid_input, -100, 100)

	# Controller inputs to tello
	tello.rc(lr=int(pid_input), fb=60)


def runPID(w, h, offset, dist_to_gate, dt, tello):
	''' Calculates error and runs the PID controller

	arguments:
	w: pixel width of the gate
	h: pixel height of the gate
	offset: pixel offset horizontally and vertically from image centre (array)
	dist_to_gate: distance to gate (cm)
	dt: step size, time since last control input (s)
	tello: drone object

	returns:
	error: error with respect to reference value (array)
	'''
	# Calculates actual distance (cm) offset from centre of gate
	# dist_offset[0]: horizontal, dist_offset[1]: vertical
	dist_offset = np.array([config.GATE_WIDTH, config.GATE_HEIGHT]) \
				/ np.array([w, h]) * offset

	# Angle to centre of gate (rads). [0]: horizontal [1]: vertical angle
	# Maybe calculating this directly from FOV is easier?
	offset_angle = np.arctan(dist_offset / dist_to_gate)

	# print('Distance horizontal: '+str(dist_offset[0])+'Distance Vertical:' \
	# 	+str(dist_offset[1]))
	# print('Horizontal angle: '+ str(offset_angle[0] * 180/np.pi) + \
	# 	'Vertical angle: '+ str(offset_angle[1] * 180/np.pi))

	# Reference for horizontal, vertical and distance
	state = tello.readState()
	reference = np.array([0, state['pitch'] - 8, 250])	# 8 degree tilt
	signal = np.concatenate((offset_angle * 180/np.pi, [dist_to_gate]))
	error = signal - reference

	try:
		PID(error, runPID.prev_error, dt, tello)
	except AttributeError:
		PID(error, error, dt, tello)

	# prev_error needs to be persistent hence is an attribute
	runPID.prev_error = error
	return error


def PID(error, prev_error, dt, tello):
	''' PID controller send commands to tello based on error. '''
	# Integral needs to be persistent hence is an attribute of PID
	try:
		PID.integral += error * dt
		#print(PID.integral)
	except AttributeError:
		PID.integral = np.zeros(len(error))

	# Numerical differentiation - first order difference scheme
	error_dot = (error - prev_error) / dt

	# PID constants and controller (Standard form)
	Kp = np.array([3.3, 6, 3.3])
	Ti = 100
	Td = np.array([0.9, 0.4, 0.7])
	pid_input = -Kp * (error + 0*PID.integral / Ti + Td * error_dot)
	#Pterm = -Kp*error
	#Iterm = -Kp/Ti * PID.integral
	#Dterm = -Kp*Td * error_dot
	#print('P term: '+str(Pterm)+'I term: '+str(Iterm)+'D term: '+str(Dterm))

	# Limit PID input to max of 100, and minimum of -100
	pid_input = controllerLimits(pid_input, -100, 100)

	# Controller inputs to tello
	tello.rc(lr=int(pid_input[0]), ud=int(pid_input[1]), fb=60)#fb=-int(0.25*pid_input[2]))


def controllerLimits(cont_input, min_limit, max_limit):
	'''
	Tello accepts a maximum of 100 and minimum of -100 for rc inputs hence this
	function prevents higher or lower values. Can also limit to smaller ranges,
	i.e. -50 to 50.

	input: Controller input terms (array)
	min_limit: Minimum controller input cutoff
	max_limit: Maximum controller input cutoff

	returns:
	limited_input: Input but with enforced limits
	'''
	limited_input = np.where(cont_input > max_limit, max_limit, cont_input)
	limited_input = np.where(limited_input < min_limit, min_limit, limited_input)
	return limited_input
