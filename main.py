


import os
import sys
import time
import socket
import cv2
import json
import time


import logging as log
import paho.mqtt.client as mqtt

import argparse
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 120


def build_argparser():
    
    
        
    parser = argparse.ArgumentParser("People Counter App")
    m_desc = "Path to an xml file with a trained model."
    i_desc = "Path to image or video file"
    l_desc = "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl."
    d_desc = "Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)"
    pt_desc = "Probability threshold for detections filtering (0.5 by default)"
    
     # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    
    required.add_argument("-m", required=True, type=str,help=m_desc)
    required.add_argument("-i", required=True, type=str, help=i_desc)
    optional.add_argument("-l", required=False, type=str, default=None, help=l_desc)
    optional.add_argument("-d", type=str, default="CPU", help=d_desc)
    optional.add_argument("-pt", type=float, default=0.5, help=pt_desc)
    args = parser.parse_args()
    return args
 

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def draw_boxes(frame, result,probability,color, width, height):
    count = 0
    for box in result[0][0]:
        conf = box[2]
        output_class = box[1]
        if conf>=probability and output_class==1 : #output_class  = 1 is for person, as per coco.names file
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            count = count+1
    return frame,count
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # Initializing Variables
    global width,height,prob_threshold
    image_mode = False
    last_count = 0
    total_count = 0
    start_time = 0
    last_last_count = 0
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.pt

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.m, args.d, args.l)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    
    # Checking Image File
    if args.i.endswith('.jpg') or args.i.endswith('.png'):
        image_mode = True
        Input_stream = args.i
    
    # Checking for input video from the camera
    elif args.i == 'CAM':
        Input_stream = 0
        
    # Checking for Video File   
    else:
        Input_stream = args.i
     
    
    cap = cv2.VideoCapture(Input_stream)
    
    if Input_stream:
        cap.open(args.i)

    if not cap.isOpened():
        log.error("ERROR! Unable to open the media")
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag,frame = cap.read()
        if not flag:
            break
        inf_start = time.time()
        
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)

        ### TODO: Wait for the result ###
        status = infer_network.wait()
        if status==0:
            det_time = time.time() - inf_start
        ### TODO: Get the results of the inference request ###
        ### TODO: Extract any desired stats from the results ###
            result = infer_network.get_output()
            frame,current_count = draw_boxes(frame, result,prob_threshold,(0,0,255), width, height)
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
        ### TODO: Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###
            
            if current_count > last_count:
                start_time = time.time()
                
            if current_count < last_count: 
                if(last_last_count>=last_count): #To avoid False Negatives
                    duration = int(time.time() - start_time)
                    total_count = total_count + last_count - current_count
                    client.publish("person", json.dumps({"total": total_count}))
                    client.publish("person/duration",json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_last_count = last_count
            last_count = current_count
            
        if key_pressed==27:
            break
        
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        if image_mode:
            cv2.imwrite('output_image.jpg', frame)
    
  
        
        ### TODO: Write an output image if `single_image_mode` ###
        
        
    
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    
        

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser()
    
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
