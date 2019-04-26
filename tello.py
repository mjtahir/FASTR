import socket
import threading
import time
import cv2 as cv
# import libh264decoder
# import numpy as np

# msg_initiate = 'command'

# # Define constants and message to start Tello's SDK
# UDP_IP_ADDRESS = '192.168.10.1'
# UDP_PORT_COMMAND = 8889
# ADDRESS_COMMAND = (UDP_IP_ADDRESS, UDP_PORT_COMMAND)
# # Create the socket. AF_NET specifies IPv4, SOCK_DGRAM specifies UDP
# client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# # Encode message and send to drone
# msg_initiate = msg_initiate.encode(encoding = 'utf-8')
# client_sock.sendto(msg_initiate, ADDRESS_COMMAND)


# ############
# state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# # Port for receiving state of Tello
# UDP_IP_SERVER = '0.0.0.0'
# UDP_PORT_RX_STATE = 8890

# # --> This should be running on a thread <--
# # Allow connections from all IP's (0.0.0.0) on a given port
# def udpReceiver():
# 	while True:
# 		try:
# 			data, addr = state_sock.recvfrom(1024) # buffer size is 1024 bytes
# 			print("received message:", data)
# 		except exception as error:
# 			print(error)
# 			break

# state_sock.bind((UDP_IP_SERVER, UDP_PORT_RX_STATE))
# udpReceiver()


# ############
# UDP_PORT_RX_VIDEO = 11111

# def turnVideoOn():
# 	video = True
# 	msg_video = 'streamon'.encode(encoding = 'utf-8')
# 	client_sock.sendto(msg_video, ADDRESS_COMMAND)

# def turnVideoOff():
# 	video = False
# 	msg_video = 'streamoff'.encode(encoding = 'utf-8')
# 	client_sock.sendto(msg_video, ADDRESS_COMMAND)

##############################################################################

class Tello:

	def __init__(self):

		# Initialise variables
		self.response = None
		self.cap = None		# Video frame
		self.background_frame_read = None
		self.command_timeout = 0.3	# Time for response (secs)

		# Address based on IP and port
		self.TELLO_IP = '192.168.10.1'
		self.TELLO_PORT = 8889
		self.TELLO_ADDRESS = (self.TELLO_IP, self.TELLO_PORT)

		# Socket for commands and video. IPv4, with UDP
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.socket_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

		# Bind the local address to enable communication
		self.LOCAL_IP = '0.0.0.0'
		self.LOCAL_PORT = 8889
		self.LOCAL_ADDRESS = (self.LOCAL_IP, self.LOCAL_PORT)
		self.socket.bind(self.LOCAL_ADDRESS)

		# Recevier thread - Daemon True so it closes when main thread ends
		self.receive_thread = threading.Thread(target=self._udpReceive, 
			daemon = True)
		self.receive_thread.start()

		# Send a byte string "command" to initiate Tello's SDK mode
		# Send a byte string "streamon" to initiate Tello's camera stream
		self.socket.sendto(b'command', self.TELLO_ADDRESS)
		print('Sent: command')
		self.socket.sendto(b'streamon', self.TELLO_ADDRESS)
		print('Sent: streamon')

		# Video port for receiving
		self.TELLO_VIDEO_PORT = 11111
		self.VIDEO_ADDRESS = (self.LOCAL_IP, self.TELLO_VIDEO_PORT)
		self.socket_video.bind(self.VIDEO_ADDRESS)
		# self.get_video_capture()
		self.get_frame_read()
		# Video receiver thread - Daemon true so it ends when main ends
		

	def _udpReceive(self):
		''' Method runs as a thread to constantly receive responses '''
		while True:
			try:
				self.response, addr = self.socket.recvfrom(2048)
				print(self.response)
			except socket.error as exc:
				print('Exception socket.error: ', exc)
	
	def closeSockets(self):
		''' Close all sockets '''
		self.socket.close()
		self.socket_video.close()

##### From: https://github.com/damiafuentes/DJITelloPy/blob/master/djitellopy/tello.py#L48
	def get_udp_video_address(self):
		return 'udp://@' + self.LOCAL_IP + ':' + str(self.TELLO_VIDEO_PORT)  # + '?overrun_nonfatal=1&fifo_size=5000'

	def get_video_capture(self):
		"""Get the VideoCapture object from the camera drone
		Returns:
			VideoCapture
		"""
		if self.cap is None:
			self.cap = cv.VideoCapture(self.get_udp_video_address())

		if not self.cap.isOpened():
			self.cap.open(self.get_udp_video_address())

		return self.cap

	def get_frame_read(self):
		"""Get the BackgroundFrameRead object from the camera drone. Then, you just need to call
		backgroundFrameRead.frame to get the actual frame received by the drone.
		Returns:
			BackgroundFrameRead
		"""
		if self.background_frame_read is None:
			self.background_frame_read = BackgroundFrameRead(self, self.get_udp_video_address()).start()
		return self.background_frame_read

		def stop_video_capture(self):
			self.socket.sendto(b'streamoff', self.TELLO_ADDRESS)
			print('Sent: streamoff')


class BackgroundFrameRead:
	"""
	This class read frames from a VideoCapture in background. Then, just call backgroundFrameRead.frame to get the
	actual one.
	"""

	def __init__(self, tello, address):
		tello.cap = cv.VideoCapture(address)
		self.cap = tello.cap

		if not self.cap.isOpened():
			self.cap.open(address)

		self.grabbed, self.frame = self.cap.read()
		self.stopped = False

	def start(self):
		self.receive_thread(target=self.update_frame, args=()).start()
		return self

	def update_frame(self):
		while not self.stopped:
			if not self.grabbed or not self.cap.isOpened():
				self.stop()
			else:
				(self.grabbed, self.frame) = self.cap.read()

	def stop(self):
		self.stopped = True
############################


	# def _videoReceive(self):
	# 	''' Method runs as a thread to constantly receive video frames '''
	# 	
	
	def sendCommand(self, command):
		print('--> Command sent:', command)
		self.socket.sendto(command.encode('utf8'), self.TELLO_ADDRESS)

		# Set abort_flag to False and create a timer thread which runs after
		# command_timeout seconds and sets abort_flag to True
		self.abort_flag = False
		timer = threading.Timer(self.command_timeout, self.setAbortFlag)
		self.socket.sendto(command.encode('utf-8'), self.tello_address)

		# Start the timer and continually check for response from Tello
		timer.start()
		while self.response is None:
			if self.abort_flag is True:
				break
		timer.cancel()
		
		if self.response is None:
			response = 'none_response'
		else:
			response = self.response.decode('utf-8')

		# Reset self.response to None and return the recorded response
		self.response = None
		return response

	def setAbortFlag(self):
		''' Sets the abort_flag to True utilised in sendCommand()'''
		self.abort_flag = True

	# Control Commands
	def takeoff(self):
		return self.sendCommand('takeoff')

	def land(self):
		return self.sendCommand('land')

	def emergency(self):
		return self.sendCommand('emergency')

	def move(self, direction, distance):
		''' Move directions include up, down, left right, forward, back
		 and distance ranges from 20 - 500 cm
		'''
		return self.sendCommand(direction + ' ' + str(distance))

	def rotate(self, direction, degrees):
		''' Rotates cw or ccw for a maximum of 3600 degrees '''
		return self.sendCommand(direction + ' ' + str(degrees))

	# Set Commands
	def setSpeed(self, speed):
		''' Sets speed given in cm/s. Range from 10 - 100 cm/s '''
		return self.sendCommand('speed '+ str(speed))

	# Read Commands
	def getSpeed(self):
		''' Returns speed in cm/s '''
		return self.sendCommand('speed?')

	def getBattery(self):
		''' Returns battery percentage '''
		return self.sendCommand('battery?')

	def getTime(self):
		''' Returns flight time in seconds '''
		return self.sendCommand('time?')

	def getHeight(self):
		''' Returns height in cm (from starting point?) '''
		return self.sendCommand('height?')

	def getTemp(self):
		''' Returns temperature in Celcius '''
		return self.sendCommand('temp?')

	def getAttitude(self):
		''' Returns IMU attitude of pitch, roll and yaw '''
		return self.sendCommand('attitude?')

	def getBaro(self):
		''' Returns barometer value (altitude?) in metres '''
		return self.sendCommand('baro?')

	def getAcceleration(self):
		''' Returns IMU acceleration in x, y and z directions in g's '''
		return self.sendCommand('acceleration?')

	def getToF(self):
		''' Returns distance from the bottom mounted 'Time-of-Flight Camera'
		sensor in cm '''
		return self.sendCommand('tof?')

	def getSNR(self):
		''' Returns Signal to Noise Ratio of the WiFi link '''
		return self.sendCommand('wifi?')



if __name__ == "__main__":
	drone = Tello()