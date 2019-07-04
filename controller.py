import numpy as np
import config
from tello_methods import Tello

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
		# constant error is set to force the drone to move the other way
		if hori_offset <= 0:
			error = -3
		else:
			error = 3

	else:
		print('No Edges found')
		tello.rc()
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
	Td = 0.5
	pid_input = -Kp * (error + Td * error_dot)

	# Limit PID input to max of 100, and minimum of -100
	pid_input = controllerLimits(pid_input, -100, 100)

	# Controller inputs to tello
	tello.rc(lr=int(pid_input))

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
	reference = np.array([0, 0, 250])
	signal = np.concatenate((offset_angle * 180/np.pi, [dist_to_gate]))
	error = signal - reference

	try:
		PID(error, runPID.prev_error, dt, tello)
	except AttributeError:
		print('run: attribute error.')
		PID(error, error, dt, tello)

	# prev_error needs to be persistent hence is an attribute
	runPID.prev_error = error
	return error


def PID(error, prev_error, dt, tello):
	''' PID controller send commands to tello based on error. '''
	# Integral needs to be persistent hence is an attribute of PID
	try:
		PID.integral += error * dt
	except AttributeError:
		PID.integral = np.zeros(len(error))
		print('PID integral initialised = ' + str(PID.integral))

	# Numerical differentiation - first order difference scheme
	error_dot = (error - prev_error) / dt

	# PID constants and controller
	Kp = 3
	Ti = 1000000000000
	Td = 0.5
	pid_input = -Kp * (error + 0*PID.integral / Ti + Td * error_dot)
	#Pterm = -Kp*error
	#Iterm = -Kp/Ti * PID.integral
	#Dterm = -Kp*Td * error_dot
	#print('P term: '+str(Pterm)+'I term: '+str(Iterm)+'D term: '+str(Dterm))

	# Limit PID input to max of 100, and minimum of -100
	pid_input = controllerLimits(pid_input, -100, 100)

	# Controller inputs to tello
	tello.rc(lr=int(pid_input[0]), ud=int(pid_input[1]), fb=-int(0.25*pid_input[2]))


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
