#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:45:55 2018

@author: amit

Main File

Example Run:
    pythonw main.py --port 10000 --trainset SumProfiles_1800s.Electricity.csv
"""
import prediction
import argparse
import socket
import json
import traceback

parser = argparse.ArgumentParser(description='Energy Consumption Prediction')

# Command Line Args
parser.add_argument('--host', type=int, required=True, help='Host on which this service will run.')
parser.add_argument('--port', type=int, required=True, help='Port on which this service will run.')
parser.add_argument('--parentdir', required=True, help='Path to parent directory of all necessary files.')
parser.add_argument('--model', required=True, help='Model name.')

args = parser.parse_args()

# Train the model initially
ecp = prediction.EnergyConsumptionPrediction(args.parentdir, args.model)
keepRunning = True

# Create a server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# get local machine name
server_socket.bind(args.host, args.port)

# queue up to 5 requests
server_socket.listen(5)

# Host a server
print("Instantiating server socket on port " + str(args.port) + " ...")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.start()

# Infinitely keep listening to messages
while keepRunning:
    try:
        # establish a connection
        clientsocket, addr = server_socket.accept()
        
        print("Got a connection from %s" % str(addr))
        msg = clientsocket.recv(4096)
        recvd_msg = json.loads(msg)
        
        predicted_value = ecp.predict(recvd_msg['timestamp'])
        clientsocket.send(json.dumps({'value':predicted_value}).encode('utf-8'))
    
    except Exception as ex:
        traceback.print_tb(ex.__traceback__)
        print("ERROR: Exception caught on server")
        clientsocket.send(json.dumps({'error':'INTERNAL SERVER ERROR'}).encode('utf-8'))
    
    finally:
        clientsocket.close()
                