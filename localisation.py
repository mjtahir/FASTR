import time
import numpy as np
from tello_methods import Tello
from NatNetClient import NatNetClient

def startTracking():
	'''
	Initiate the OptiTrack (streamingClient) thread by constructing the object 
	and running it. This is based on the NatNet v3.1 code supplied through 
	NaturalPoint's SDK online.
	'''
	# This will create a new NatNet client
	streamingClient = NatNetClient()

	# Configure the streaming client to call our rigid body handler on the 
	# emulator to send data out.
	streamingClient.newFrameListener = True
	streamingClient.rigidBodyListener = True

	# Start up the streaming client now that the callbacks are set up.
	# This will run perpetually, and operate on a separate thread.
	streamingClient.run()
	# Slow the OptiTrack threads down?

	# Time to retrieve first state. If no state currently received, the listener
	# just returns True.
	start_time = time.time()
	print('Connecting to OptiTrack .....')
	while streamingClient.rigidBodyListener is True:
		
		current_time = time.time()
		elapsed_time = current_time - start_time
		if elapsed_time > 3:
			print('Did not receive data from OptiTrack')
			return False

	return streamingClient


def trueState(streamingClient):
	'''
	Retrieves the latest true rigid body state. Note that only one rigid body 
	should be selected in Motive as this function ignores the ID number. The 
	position and orientation is changed to a standard coordinate system. The 
	orientation is returned in Euler angles.
	'''
	# Retrieve rigid body data from OptiTrack.
	id_num, pos, quaternion = streamingClient.rigidBodyListener

	# Rotate coordinates to aircraft standard (forward x, right y, down z) from
	# left x, back y, up z.
	#rotated_pos = Cz(np.pi/2) @ Cx(np.pi/2) @ streamingClient.rigidBodyListener[1]
	rotated_pos = Cx(np.pi) @ Cz(-np.pi/2) @ pos
	rotated_pos = rotated_pos * 100		# change to cm

	rotated_quat = Cx(np.pi) @ Cz(-np.pi/2) @ quaternion[0:3]
	rotated_quat = np.concatenate([rotated_quat, np.array([quaternion[3]])])
	euler = quaternion2Euler(rotated_quat)
	
	return np.array([id_num, rotated_pos, rotated_quat, euler])


def waypointGeneration(streamingClient):
	'''
	Generate your waypoints here by manually specifying or use a mathematical
	function to generate a shape. Note that the waypoints can be made relative 
	to other waypoints which avoids the use of the absolute (world) frame of 
	reference. The controller defaults back to the first waypoint (waypoint[0]) 
	upon reaching the final waypoint (see waypointUpdate()).
	'''
	# theta = np.linspace(0, 2*np.pi, 100)
	# radius = 150	# cm
	# x = radius * np.sin(theta)
	# y = radius * np.cos(theta)
	# z = np.ones(100) * -200

	# waypoint = np.zeros([100, 3])
	# waypoint[:, 0] = x
	# waypoint[:, 1] = y
	# waypoint[:, 2] = z

	start_position = trueState(streamingClient)[1]

	# State number of waypoints and list them below. Note units are in cm and 
	# order is (x,y,z) in the standard aircraft coordinate system.
	num_of_waypoints = 4
	waypoint = np.zeros([num_of_waypoints, 3])
	waypoint[0] = start_position
	waypoint[1] = waypoint[0] + np.array([600, 0, 0])
	waypoint[2] = waypoint[1] + np.array([0, 300, 0])
	waypoint[3] = waypoint[2] + np.array([-600, 0, 0])
	# waypoint[4] = waypoint[3] + np.array([0, -300, 0])
	
	return waypoint


def waypointUpdate(streamingClient, waypoint):
	'''
	Keeps index of which waypoint to track. The waypoint switching is 
	based on distance so it switches to the next when within a certain range 
	of the current.
	'''
	# Attain current position (x,y,z)
	true_state = trueState(streamingClient)
	current_position = true_state[1]

	try:
		# Relative vector from current position to current waypoint
		r_wd = waypoint[waypointUpdate.current_waypoint] - current_position
	except (IndexError, AttributeError):
		# First time run or all waypoints reached therefore set to start
		waypointUpdate.current_waypoint = 0
		r_wd = waypoint[waypointUpdate.current_waypoint] - current_position

	# Distance to the next waypoint. Transition to next if within 50 cm.
	distance_to_waypoint = np.linalg.norm(r_wd)
	if distance_to_waypoint < 50:
		waypointUpdate.current_waypoint += 1
		# r_wd will  be updated in next iteration to avoid IndexError's.

	return r_wd


def quaternion2Euler(quat):
	'''
	Converts Quaternions to Euler Angles
	Input: vector 'quat' in [q_x, q_y, q_z, q_w] format
	Output: vector [phi,theta,psi]' in radians
	'''
	# Separate variables for the quaternions
	q0 = quat[3]
	q1 = quat[0]
	q2 = quat[1]
	q3 = quat[2]

	# Calculate the Euler Angles
	theta = np. arctan2(q0 * q2 - q1 * q3,
		np.sqrt((q0 ** 2 + q1 ** 2 - 0.5) ** 2 + (q1 * q2 + q0 * q3) ** 2))
	phi = np. arctan2(q2 * q3 + q0 * q1, q0 ** 2 + q3 ** 2 - 0.5)
	psi = np. arctan2(q1 * q2 + q0 * q3, q0 ** 2 + q1 ** 2 - 0.5)

	# Construct the return array
	euler = np.array([phi, theta, psi])
	return euler


def Cx(angle):
	'''
	DCM for rotation about x axis
	Input: rotation angle in radians
	Output: DCM
	'''
	# Rotation matrix
	rotate_x = np.array([1, 0, 0,
		0, np.cos(angle), np.sin(angle),
		0, -np.sin(angle), np.cos(angle)])

	# Shape correctly to 3 x 3
	rotate_x.shape = (3, 3)

	return rotate_x


def Cy(angle):
	'''
	DCM for rotation about y axis
	Input: rotation angle in radians
	Output: DCM
	'''
	# Rotation matrix
	rotate_y = np.array([np.cos(angle), 0, -np.sin(angle),
			0, 1, 0,
			np.sin(angle), 0, np.cos(angle)])

	# Shape correctly to 3 x 3
	rotate_y.shape = (3, 3)

	return rotate_y


def Cz(angle):
	'''
	DCM for rotation about z axis
	Input: rotation angle in radians
	Output: DCM
	'''
	# Rotation matrix
	rotate_z = np.array([np.cos(angle), np.sin(angle), 0,
		-np.sin(angle), np.cos(angle), 0,
		0, 0, 1])

	# Shape correctly to 3 x 3
	rotate_z.shape = (3, 3)

	return rotate_z
