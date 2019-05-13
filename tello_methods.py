import socket
import threading
import time
import cv2 as cv
# import libh264decoder
# import numpy as np

class Tello:

	def __init__(self):

		# Initialise variables
		self.response = None
		self.cap = None		# Video object
		self.frame = None   # Video frame
		self.command_timeout = 0.3	# Time for response (secs)

		# Address based on IP and port
		self.TELLO_IP = '192.168.10.1'
		self.TELLO_PORT = 8889
		self.TELLO_ADDRESS = (self.TELLO_IP, self.TELLO_PORT)

		# Socket for commands and video. IPv4, with UDP
		self.socket_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.socket_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

		# Bind the local address to enable communication
		self.LOCAL_IP = '0.0.0.0'
		self.LOCAL_PORT = 8889
		self.LOCAL_ADDRESS = (self.LOCAL_IP, self.LOCAL_PORT)
		self.socket_cmd.bind(self.LOCAL_ADDRESS)

		# Recevier thread - Daemon True so it closes when main thread ends
		self.receive_thread = threading.Thread(target=self._udpReceive, 
			daemon = True)
		self.receive_thread.start()

		# Send a byte string "command" to initiate Tello's SDK mode
		self.socket_cmd.sendto(b'command', self.TELLO_ADDRESS)
		print('Sent: command')

		# Video port for receiving
		self.TELLO_VIDEO_PORT = 11111
		self.VIDEO_ADDRESS = (self.LOCAL_IP, self.TELLO_VIDEO_PORT)
		# self.socket_video.bind(self.VIDEO_ADDRESS)

	def _udpReceive(self):
		''' Method runs as a thread to constantly receive responses '''
		while True:
			try:
				self.response, addr = self.socket_cmd.recvfrom(2048)
				print(self.response.decode('utf-8'))
			except Exception as exc:
				print('Exception in udpReceiver. socket.error:', exc)
	
	def startVideoCapture(self):
		''' Initiates video capture by starting the Tello camera, finding the 
		pointer to the video (VideoCapture) and starting the thread which reads
		each frame
		 '''
		# Send a byte string "streamon" to initiate Tello's camera stream
		self.socket_cmd.sendto(b'streamon', self.TELLO_ADDRESS)
		print('Sent: streamon')

		# Retreive pointer to camera. The address was found online
		# Then start threading
		self.cap = cv.VideoCapture('udp://@' + self.LOCAL_IP + ':' 
					+ str(self.TELLO_VIDEO_PORT)) #'?overrun_nonfatal=1&fifo_size=5000'
		self.video_thread = threading.Thread(target=self._updateFrame, 
							daemon=True)
		self.video_thread.start()
		time.sleep(1)		# Initial frames takes time

	def _updateFrame(self):
		''' Updates frame through a thread '''
		while True:
			try:
				self.ret, self.frame = self.cap.read()
			except Exception as exc:
				print('Exception in updateFrame. cap.read:', exc)

	def readFrame(self):
		''' Returns frame to the calling object '''
		return self.frame

	def shutdown(self):
		''' Shutdown procedure, stop video capture and close sockets '''
		self.stopVideoCapture()
		self.socket_cmd.close()
		self.socket_video.close()

	def stopVideoCapture(self):
		self.socket_cmd.sendto(b'streamoff', self.TELLO_ADDRESS)
		print('Sent: streamoff')
		self.cap.release()

	def sendCommand(self, command):
		''' Sends utf8 encoded command to the Tello and sleeps before sending
		next command to allow time for a response.
		'''
		print('--> Command sent:', command)
		self.socket_cmd.sendto(command.encode('utf8'), self.TELLO_ADDRESS)
		time.sleep(self.command_timeout)
		# Set abort_flag to False and create a timer thread which runs after
		# command_timeout seconds and sets abort_flag to True
		# self.abort_flag = False
		# timer = threading.Timer(self.command_timeout, self._setAbortFlag)
		# self.socket_cmd.sendto(command.encode('utf-8'), self.TELLO_ADDRESS)

		# # Start the timer and continually check for response from Tello
		# timer.start()
		# while self.response is None:
		# 	if self.abort_flag is True:
		# 		break
		# timer.cancel()
		
		# if self.response is None:
		# 	response = 'none_response'
		# else:
		# 	response = self.response.decode('utf-8')

		# # Reset self.response to None and return the recorded response
		# self.response = None
		# return response

	# def _setAbortFlag(self):
	# 	''' Sets the abort_flag to True utilised in sendCommand() '''
	# 	self.abort_flag = True

	def sendCommandNoResponse(self, command):
		''' Sends utf8 encoded command to the Tello '''
		print('--> Command sent:', command)
		self.socket_cmd.sendto(command.encode('utf8'), self.TELLO_ADDRESS)

	# Control Commands
	def takeoff(self):
		''' Sends takeoff command '''
		return self.sendCommand('takeoff')

	def land(self):
		''' Sends land command '''
		return self.sendCommand('land')

	def emergency(self):
		''' Stops all motors immediately '''
		return self.sendCommand('emergency')

	def move(self, direction, distance):
		''' Move directions include up, down, left, right, forward, back
		 and distance ranges from 20 - 500 cm
		'''	
		return self.sendCommand(direction + ' ' + str(distance))

	def rotate(self, direction, degrees):
		''' Rotates cw or ccw for a maximum of 3600 degrees '''
		return self.sendCommand(direction + ' ' + str(degrees))

	# Set Commands
	def setSpeed(self, speed):
		''' Sets speed given in cm/s. Range from 10 - 100 cm/s '''
		return self.sendCommand('speed ' + str(speed))

	def rc(self, lr=0, fb=0, ud=0, yaw=0):
		''' Allows for 4 channel remote controller type commands to be sent.
			The limits for each input is -100 to +100 (percent?)
		'''
		return self.sendCommandNoResponse('rc ' + str(lr) + ' ' + str(fb)
			+ ' ' + str(ud) + ' ' + str(yaw))

	# Read Commands
	def getSpeed(self):
		''' Returns speed in cm/s '''
		return self.sendCommand('speed?')

	def getBattery(self):
		''' Returns battery percentage '''
		return self.sendCommand('battery?')

	def getTime(self):
		''' Returns flight time since turned on in seconds '''
		return self.sendCommand('time?')

	def getHeight(self):
		''' Returns height in dm (decimeter) from starting point 
			This may be IMU based hence experiences drift. The return
			value is rounded to the nearest decimeter.
		'''
		return self.sendCommand('height?')

	def getTemp(self):
		''' Returns temperature in Celcius '''
		return self.sendCommand('temp?')

	def getAttitude(self):
		''' Returns IMU attitude of pitch, roll and yaw in degrees. 
			Yaw is zeroed when Tello is turned on.
		'''
		return self.sendCommand('attitude?')

	def getBaro(self):
		''' Returns barometer value (altitude) in metres.
			Seems to fluctuate a fair amount
		 '''
		return self.sendCommand('baro?')

	def getAcceleration(self):
		''' Returns IMU acceleration in x, y and z directions in
			0.001g's. So 1000 = 1g.
		 '''
		return self.sendCommand('acceleration?')

	def getToF(self):
		''' Returns distance from the bottom mounted 'Time-of-Flight Camera'
		in mm. Minimum is 100 mm and anything below 300 mm defaults to 100
		mm. Seems fast and accurate.
		'''
		return self.sendCommand('tof?')

	def getSNR(self):
		''' Returns Signal to Noise Ratio of the WiFi link '''
		return self.sendCommand('wifi?')


if __name__ == "__main__":
	drone = Tello()
	drone.startVideoCapture()
	
	while True:
		frame = drone.readFrame()
		cv.imshow('Frame', frame)
		if cv.waitKey(1) & 0xFF == ord('q'):
			drone.shutdown()
			break
	drone.getBattery()
