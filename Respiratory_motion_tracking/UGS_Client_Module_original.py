"""
============================
Simple client
============================
Simplest example of getting a transform from a device.
"""


from math import cos, sin, pi
import pyigtl  # pylint: disable=import-error
import matplotlib.pyplot as plt
#HY codes import strated here
from utils.models import RNetE2E
from utils.config_dx import *
import numpy as np
import os
from utils.RTransform import RTransform_keywords as RTransform
import cv2 as cv
from utils.transform_matrix_v5 import image_transform
from scipy import signal
import csv

import time
import sys
sys.path.append('/home/toe/UGS/UGSIntegration_Animal/')
from utils.cal_metrics import *

import tensorflow as tf

import os

import math
#HY codes impor ended here
#CYL codes import started here
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


import glob
import SimpleITK as sitk
import numpy as np

from PIL import Image

import os
import nibabel as nib
import vtk

from timeit import default_timer as timer

#!/usr/bin/env python



import json

from numba import jit
import logging

#Read the config file and prepare the folders
from configparser import ConfigParser
config_object = ConfigParser()
config_object.read("config.ini")

ip_address = config_object["SETTINGS"]["IPAddress"]
port_number = config_object["SETTINGS"]["PortNumber"]

CT_US_Registration_On_Off = config_object["SETTINGS"]["CT_US_Registration"]
US_US_Registration_On_Off = config_object["SETTINGS"]["US_US_Registration"]

PUCA_Image_Size_X = config_object["SETTINGS"]["PUCA_Image_Size_X"]
PUCA_Image_Size_Y = config_object["SETTINGS"]["PUCA_Image_Size_Y"]

from utils.probeArrayQueue import probeArrayQueue
from utils.MotionTracking import MotionTracking, MotionTrackingPhase

##UGSClient Class
class UGSClient:
    ##The constructor
    # @param self Object
    def __init__(self):
        
        
        self.client_task()
        
        

    ## This function perform presetting of the US-US registration 
    # @param img Referenced Ultrasound Image
    def presettingUSUSWithReferenceImg(self, img):

        
    #============== END of US-US Preparation =====================

    ## This function process the US-US registration 
    # @param img Realtime Ultrasound Image
    def processUSUS(self, img):
    ############### start to register ###############

        return transform_matrix
    #============== END of US-US Registration =====================

    ## This function perform CT-US registration 
    # @param img Realtime Ultrasound Image
    # @param model_ct_feature Feature Model CT
    # @param model_us_feature Feature Model US
    # @param model_registration Registration Model
    # @param RCTimg RCT Image
    # @param RCTseg RCT Segmentation
    # @param RCTfeat RCT Feature
    # @param CTimg CT Image
    # @param CTseg CT Segmentation
    # @param CTfeat CT Feature
    # @param CT_seg CT_Segmentationm
    # @param CT_Feat CT_Feature
    # @param com_slices Comm Slices
    def processCTUS(self, img, model_ct_feature, model_us_feature, model_registration, RCTimg, RCTseg, RCTfeat, CTimg, CTseg, CTfeat, CT_Seg, CT_Feat,com_slices):
        

    #============== END of CT-US Registration =====================

    ## This function preparing CT-US registration 
    def prepareCT_US(self):	
        
    #============== END of CT-US Preparation =====================

    ## This function calculating the bounding box of the kidney roi
    # @param points Points selected
    def bounding_box(self, points):
        x_coord, y_coord = zip(*points)
        return[(min(x_coord), min(y_coord), max(x_coord), max(y_coord))]

    def updateCroppingWindowParams(self, binaryInage):
        contours, hierarchy = cv.findContours(binaryInage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv.contourArea)
        moments = cv.moments(max_contour)
        self.centerX = int(moments["m10"] / moments["m00"])
        self.centerY = int(moments["m01"] / moments["m00"])
        
    def cropImage(self, img):
        new_width = 509
        new_height = 509
        tl_x = int(self.centerX - (new_width / 2))
        tl_y = int(self.centerY - (new_height / 2))
        br_x = int(self.centerX + (new_width / 2))
        br_y = int(self.centerY + (new_height / 2))
        croppedImg = img[tl_y:br_y, tl_x:br_x]
        return croppedImg;
        
    ## Main function of the UGS Client Module for Animal Data
    def client_task(self):
        motionTracker = MotionTracking();
        
        log = logging.getLogger(__name__)
        log.setLevel(logging.DEBUG)
        logging_handler = logging.FileHandler('Log.log', mode='w')
        logging_handler.setLevel(logging.DEBUG)
        logging_handler_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging_handler.setFormatter(logging_handler_format)
        log.addHandler(logging_handler)

        npArr = np.array([[[100,0,0,0]]], dtype='uint8')
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        im1 = ax1.imshow(npArr,cmap=plt.cm.gray,vmin=0,vmax=255)
        im2 = ax2.imshow(npArr,cmap=plt.cm.gray,vmin=0,vmax=255)
        plt.ion()
        
        self.probe_pose0 = np.identity(4)
        
        if self.CT_US_Registration == 1:
            #Prepare CT/US          
        if self.ip_address == "127.0.0.1": 
            self.ip_address = "localhost"    
            
        client = pyigtl.OpenIGTLinkClient(host=str(self.ip_address), port=int(self.port_number))  

        log.info('Pyigtl client is created!')
        print("IGTLink Client Setup")
        while True:
            start = timer()
            messages = client.get_latest_messages()
            for message in messages:
                #print(message)
                timestamp = message.timestamp
                timestamp_for_log = "{:.2f}".format(message.timestamp)
                print('center is at ', self.centerX)
                if message.device_name == "BinaryImage":
                    print(message);
                    #binary_img_data = np.squeeze(message.image.reshape(1,self.imageSizeX,self.imageSizeY,1).transpose(0,1,2))
                    fn = 'roi_bin.jpg'
                    save_path 	= './data/kidney_boundary/testOutput'
                    img_data2d = np.squeeze(message.image.reshape(1,self.imageSizeX,self.imageSizeY).transpose(0,1,2))
                    img_data = Image.fromarray(img_data2d)
                    img_data.save(os.path.join(save_path, fn))
 
                    self.updateCroppingWindowParams(np.asarray(img_data))
                    img_data = self.cropImage(np.asarray(img_data))
                    fn = 'roi_bin_cropped.jpg'
                    img2s = Image.fromarray(img_data)
                    img2s.save(os.path.join(save_path, fn))

                    print('new center is at ', self.centerX)
                    print('saved binary data')
                    readyToPreset = True;
                if message.device_name == "Config_JSON_Msg":
                    log.info('Timestamp: %s - Config JSON Message is received from server!', timestamp_for_log)
                    json_string = message.string
                    global config_params
                    config_params = json.loads(json_string)
                if message.device_name == 'USImageRef':
                    log.info('Timestamp: %s - Reference Image is received from server!', timestamp_for_log)
                    if self.CT_US_Registration == 1:
                        #Converting 2d image received to 3d itk image
                        img_data3d = message.image.reshape(1,self.imageSizeX,self.imageSizeY).transpose(0,1,2)
                        img_data = sitk.GetImageFromArray(img_data3d, isVector=False)
                        img_data = self.cropImage(img_data)
    
                        spacin_img_data = [float(config_params["spacing_x"]),float(config_params["spacing_y"]),float(config_params["spacing_z"])]
                        img_data.SetSpacing(spacin_img_data)
                        origin_img_data = [float(config_params["origin_x"]),float(config_params["origin_y"]),float(config_params["origin_z"])]
                        img_data.SetOrigin(origin_img_data)
    
                        #Process CT/US
                        controurArray = sitk.GetArrayFromImage(contours) 
                        #process matrix and generate transform
                        image_message = pyigtl.ImageMessage(controurArray, device_name="xContourRef", timestamp=timestamp)
                        transform_message = pyigtl.TransformMessage(US_matrix, device_name="xMatrixRef", timestamp=timestamp)
                        # Send messagestransform_message
                        client.send_message()
                        client.send_message(image_message)
                    if self.US_US_Registration == 1:
                        img = message.image.reshape(self.imageSizeX,self.imageSizeY)
                        print(message.ijk_to_world_matrix)
                        #self.presettingUSUSWithReferenceImg(img)
                        
                        #saving probe pose for frame 0, with ref image frame
                        self.probe_pose0 = message.ijk_to_world_matrix
                        
                if message.device_name == 'USImage':
                    
                    log.info('Timestamp: %s - US Image is received from server!', timestamp_for_log)
                    fn = 'ori_cropped.jpg'
                    save_path 	= './data/kidney_boundary/testOutput/'
                    imgRead = Image.open(os.path.join(save_path, fn))


                    im1.set_data(imgRead)
                    #im2.set_data(message.image.reshape(self.imageSizeX,self.imageSizeY))
                    
                    fn2 = 'reg.jpg'
                    imgRead2 = Image.open(os.path.join(save_path, fn2))
                    im2.set_data(imgRead2)
                    
                    plt.pause(0.1)
                    
                    if self.US_US_Registration == 1:
                        img = message.image.reshape(self.imageSizeX,self.imageSizeY)
                        img = self.cropImage(img)
                        img2s = Image.fromarray(img)
                        img2s.save(os.path.join(save_path, fn))
                        
                        if self.USUSPreset == False:
                            #Preset US/US
                        else:
                            #print("crop_rect is ", crop_rect)
                            #img = crop_image(img, mask, crop_rect)
                            startUSUS = timer()
                            transform_matrix = #ProcessUSUS, get the matrix
                            print('Time to process one registration', timer() - startUSUS)
                            # Generate transform
        
                            transform_message = pyigtl.TransformMessage(transform_matrix, device_name="xMatrixUSUS", timestamp=timestamp)
                            # Send messages    
                            client.send_message(transform_message)
                            print(transform_message)
                            log.info('Timestamp: %s - US-US Registration done! Transformation Matrix sent back!', timestamp_for_log)
                            py, pz = motionTracker.preprocessing_data(self.probe_pose0, message.ijk_to_world_matrix, transform_matrix)  
                            self.probeArray.add_item([[py, pz, timestamp]])
                            print("probeArray's length is ", self.probeArray.get_num_element())
                            if self.probeArray.is_queue_full():
                                yy, zz, tt = self.probeArray.get_yy_zz_tt()
                                motionModelParams = motionTracker.fit_sin(tt, zz, yy)
                                json_text_to_parse = str(motionModelParams["amp"]) + " " + str(motionModelParams["period"]) + " " + str(motionModelParams["phase"]) + " " + str(motionModelParams["offset"]) + " " + str(motionModelParams["ny"]) + " " + str(motionModelParams["nz"])
                                #json_text_to_parse = '{ "amp":"'+str(motionModelParams["amp"])+'", "period":"'+str(motionModelParams["period"])+'", "phase":"'+str(motionModelParams["phase"])+'", "offset":"'+str(motionModelParams["offset"])+'", "ny":"'+str(motionModelParams["ny"])+'", "nz":"'+str(motionModelParams["nz"])+'"}'
                                modelParamMessage = pyigtl.StringMessage(json_text_to_parse, device_name="MotionModelParam", timestamp=timestamp)
                                client.send_message(modelParamMessage, wait=True)
                                print("MotionModelParam is sent ", json_text_to_parse)

                    if self.CT_US_Registration == 1:
                        
                    #print(message)
                if message.device_name == 'String':
                    print(message)
                print('\n')   
                
                print('Time to process one loop', timer() - start)
        # Send messages
        client.send_message(string_message)
        
