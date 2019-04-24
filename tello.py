import socket
import time

# Define constants and message to start Tello's SDK
UDP_IP_ADDRESS = '192.168.10.1'
UDP_PORT_COMMAND = 8889
ADDRESS_COMMAND = (UDP_IP_ADDRESS, UDP_PORT_COMMAND)
msg_initiate = 'command'

# Create the socket. AF_NET specifies IPv4, SOCK_DGRAM specifies UDP
client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Encode message and send to drone
msg_initiate = msg_initiate.encode(encoding = 'utf-8')
client_sock.sendto(msg_initiate, ADDRESS_COMMAND)


############
state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Port for receiving state of Tello
UDP_IP_SERVER = '0.0.0.0'
UDP_PORT_RX_STATE = 8890

# --> This should be running on a thread <--
# Allow connections from all IP's (0.0.0.0) on a given port
def udpReceiver():
	while True:
		try:
			data, addr = state_sock.recvfrom(1024) # buffer size is 1024 bytes
			print("received message:", data)
		except exception as error:
			print(error)
			break

state_sock.bind((UDP_IP_SERVER, UDP_PORT_RX_STATE))
udpReceiver()


############
UDP_PORT_RX_VIDEO = 11111

def turnVideoOn():
	video = True
	msg_video = 'streamon'.encode(encoding = 'utf-8')
	client_sock.sendto(msg_video, ADDRESS_COMMAND)

def turnVideoOff():
	video = False
	msg_video = 'streamoff'.encode(encoding = 'utf-8')
	client_sock.sendto(msg_video, ADDRESS_COMMAND)