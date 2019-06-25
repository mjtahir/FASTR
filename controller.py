import numpy as np
import config
from tello_methods import Tello

def run(w, h, offset, dist_to_gate, dt, tello):
	if w is not None:

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

		#tello.rc(lr=int(epsilon_x * 180/np.pi)*4)
		#tello.rc(ud=-int(epsilon_y * 180/np.pi)*4)
		error = offset_angle * 180/np.pi
		try:
			PID(error, run.prev_error, dt, tello)
		except AttributeError:
			print('run: attribute error.')
			PID(error, error, dt, tello)

		run.prev_error = error
		return error
	else:
		tello.rc()

# Test each 3 controller terms separately

def PID(error, prev_error, dt, tello):

	# Integral needs to be persistent hence is an attribute of PID
	try:
		PID.integral += error * dt
	except AttributeError:
		PID.integral = np.zeros(2)
		print('PID integral initialised = [0,0]')

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
	pid_input = np.where(pid_input > 100, 100, pid_input)
	pid_input = np.where(pid_input < -100, -100, pid_input)

	# Controller inputs to tello (check sign)
	tello.rc(lr=int(pid_input[0]), ud=int(pid_input[1]))

	#test, out of batteries