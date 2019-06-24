import numpy as np
import config

def run(w, h, offset, dist_to_gate, prev_error, dt):
	if w is not None:
		# hori_offset = 50/w * offset[0]
		# vert_offset = 50/h * offset[1]

		# Calculates actual distance (cm) offset from centre of gate
		# dist_offset[0]: horizontal, dist_offset[1]: vertical
		dist_offset = np.array([config.GATE_WIDTH, config.GATE_HEIGHT]) \
					/ np.array([w, h]) * offset

		# epsilon_x = np.arctan(hori_offset/distance)
		# epsilon_y = np.arctan(vert_offset/distance)

		# Angle to centre of gate (rads). [0]: horizontal [1]: vertical angle
		offset_angle = np.arctan(dist_offset / dist_to_gate)

		#print('Horizontal angle: '+ str(epsilon_x * 180/np.pi))
		#print('Vertical angle: '+ str(epsilon_y * 180/np.pi))

		#tello.rc(lr=int(epsilon_x * 180/np.pi)*4)
		#tello.rc(ud=-int(epsilon_y * 180/np.pi)*4)
		error = offset_angle * 180/np.pi
		PID(error, prev_error, dt)

		return error
	else:
		tello.rc()

# Check dist_offset numbers, see if they are realistic at different distances.
# Test each 3 controller terms separately

def PID(error, prev_error, dt):

	# Integral needs to be persistent hence is an attribute of PID
	try:
		PID.integral += error*dt
	except AttributeError:
		PID.integral = np.zeros(2)
		print('PID integral initialised = [0,0]')

	# Numerical differentiation - first order difference scheme
	error_dot = (error - prev_error) / dt

	# PID constants and controller
	Kp = 1
	Ti = 1
	Td = 1
	pid_input = Kp * (error + PID.integral / Ti + Td * error_dot)

	# Controller inputs to tello (check sign)
	tello.rc(lr=int(pid_input[0]), ud=-int(pid_input[1]))