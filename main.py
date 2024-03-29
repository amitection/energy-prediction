#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:45:55 2018

@author: amit

Main File

Example Run:
    python main.py --host 35.231.103.50 --port 10000 --parentdir house1 --model house1
"""
import consumption.prediction as consumption_prediction
import argparse
import socket
import simplejson as json
import traceback
import constants

parser = argparse.ArgumentParser(description='Energy Consumption Prediction')

# Command Line Args
parser.add_argument('--host', required=True, help='Host on which this service will run.')
parser.add_argument('--port', type=int, required=True, help='Port on which this service will run.')
parser.add_argument('--parentdir', required=True, help='Path to parent directory of all necessary files.')
parser.add_argument('--model', required=True, help='Model name.')

args = parser.parse_args()

# Train the model initially
ecp = consumption_prediction.EnergyConsumptionPrediction(args.parentdir, args.model)
keepRunning = True

# Create a server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# get local machine name
server_socket.bind((socket.gethostname(), args.port))

print("Instantiating server socket on port " + str(args.port) + " ...")

# Host a server
# queue up to 5 requests
server_socket.listen(5)

def initiate_server(keepRunning):
    # Infinitely keep listening to messages
    try:
        while keepRunning:
            # establish a connection
            clientsocket, addr = server_socket.accept()
            
            try:
                print("Got a connection from %s" % str(addr))
                msg = clientsocket.recv(4096)
                print('Ori message: '+str(msg))
                recvd_msg = json.loads(msg)
                
                # If message topic incorrect 
                if ('topic' not in recvd_msg) or (recvd_msg['topic'] not in constants.topic):
                    clientsocket.send(json.dumps({'error':constants.satus['400']}).encode('utf-8'))
                
                elif recvd_msg['topic'] == 'EXIT':
                    keepRunning = False
                    clientsocket.send(json.dumps({'message':'Exiting!'}).encode('utf-8'))
                else:
                    message = message_handler(recvd_msg)
                    clientsocket.send(message)
                
                
            
            except Exception:
                print(traceback.format_exc())
                print("ERROR: Exception caught on server")
                clientsocket.send(json.dumps({'error':constants.satus['500']}).encode('utf-8'))
            
            finally:
                clientsocket.close()
    finally:
        print("Shutdown hook initiated..")
        
        print("Closing server socket safely.")
        # Close the server socket
        server_socket.close()


def message_handler(recvd_msg):
    if recvd_msg['topic'] == 'FORECAST_CONSUMPTION':
            return forecast_consumption(recvd_msg)
    elif recvd_msg['topic'] == 'FORECAST_GENERATION':
            return forecast_generation(recvd_msg)
        
def forecast_consumption(recvd_msg):
    # message contains the timestamp
    predicted_value = ecp.predict(recvd_msg['message'])
    message = json.dumps({'message':predicted_value}).encode('utf-8')
    return message

def forecast_generation(recvd_msg):
    # message contains the timestamp
    print('Implementation TBD')
    # TODO
    message = json.dumps({'message':'To be implemented'}).encode('utf-8')
    return message


# Start the server
initiate_server(keepRunning)



