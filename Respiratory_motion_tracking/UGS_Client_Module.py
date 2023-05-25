"""
============================
Simple client
============================
Simplest example of getting a transform from a device.
"""

# from math import cos, sin, pi
import pyigtl  # pylint: disable=import-error
import matplotlib.pyplot as plt
#HY codes import strated here

import numpy as np
import os
import cv2 as cv
import SimpleITK as sitk
from PIL import Image
from timeit import default_timer as timer

# from utils.probeArrayQueue import probeArrayQueue
from utils.MotionTracking import MotionTracking, probeArrayQueue

import json
import logging
from configparser import ConfigParser
import pydicom

#!/usr/bin/env python


#Read the config file and prepare the folders

# config_object = ConfigParser()
# config_object.read("config.ini")

# ip_address = config_object["SETTINGS"]["IPAddress"]
# port_number = config_object["SETTINGS"]["PortNumber"]
ip_address = "127.0.0.1"
port_number = 23338

# CT_US_Registration_On_Off = config_object["SETTINGS"]["CT_US_Registration"]
# US_US_Registration_On_Off = config_object["SETTINGS"]["US_US_Registration"]

# PUCA_Image_Size_X = config_object["SETTINGS"]["PUCA_Image_Size_X"]
# PUCA_Image_Size_Y = config_object["SETTINGS"]["PUCA_Image_Size_Y"]


##UGSClient Class
class UGSClient:
    ##The constructor
    # @param self Object
    def __init__(self):
        #self.client_task()
        #self.readImageAndTrackObjects()
        self.US_US_Registration = 1
        self.m_contours = None
        self.m_US_matrix = None
        self.probeArray = probeArrayQueue()
        self.m_motionTracker = MotionTracking()
        self.folder_path = 'D:/work/Respiratory_motion_tracking/data/video recorded/Patient_-1_Surgery_-1_Timestamp_212.387'  # CHANGE TO THE CORRECT DATA FOLDER PATH
        self.m_referenceFrame = None
        self.m_processingFrame = None

        self.featureDetector = cv.ORB_create()
        self.featureMatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.reference_kp, self.reference_des = None, None
        self.matchedImage = None

        ## dicom image param
        self.pixelSpacing = None
    
    def readImageAndTrackObjects(self):
        images = []
        reference_image = None
        refImageWithRoi = None
        
        frame = 0
        file_names = [file_name for file_name in os.listdir(self.folder_path) if file_name.endswith(".dcm")]
        # Remove the extension from the file names
        file_names = [os.path.splitext(file_name)[0] for file_name in file_names]
        # Sort the file names based on the desired order
        file_names.sort(key=lambda x: int(x.split("_")[2]))

        for filename in file_names:
            frame += 1
            
            file_path = os.path.join(self.folder_path, filename + ".dcm")
            dcm = pydicom.dcmread(file_path)
            self.pixelSpacing = dcm[0x0018,0x2010][0]
            timestamp = float(dcm[0x0008,0x0030].value)
            print(f"Pixel Spacing....: {self.pixelSpacing}") ## Normalized pixel spacing

            pixel_array = dcm.pixel_array
            normalized_array = cv.normalize(pixel_array, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            image = cv.cvtColor(normalized_array, cv.COLOR_GRAY2BGR)

            # Define the rectangle area of interest (ROI)
            roi_width = 222  # width of the ROI
            roi_height = 170  # height of the ROI
            roi_x = image.shape[1]//2 - roi_width//2
            roi_y = round(image.shape[0]//2)

            if reference_image is None:
                reference_image = image
                images.append(reference_image)
                self.m_referenceFrame = reference_image
                reference_gray = cv.cvtColor(reference_image, cv.COLOR_BGR2GRAY)
                self.reference_kp, self.reference_des = self.featureDetector.detectAndCompute(reference_gray, None)

                refImageWithRoi = self.m_referenceFrame.copy()
                cv.rectangle(refImageWithRoi, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)
                self.start_time = timestamp
                
                
            else:
                images.append(image)
                new_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                new_kp, new_des = self.featureDetector.detectAndCompute(new_gray, None)
                # Match keypoints between the reference frame and the new frame
                matches = self.featureMatcher.match(self.reference_des, new_des)

                filtered_matches = []
                for match in matches:
                    new_kp_idx = match.trainIdx
                    new_keypoint = new_kp[new_kp_idx]
                    if roi_x <= new_keypoint.pt[0] <= roi_x + roi_width and roi_y <= new_keypoint.pt[1] <= roi_y + roi_height:
                        filtered_matches.append(match)

                ##### Sort matches by distance
                filtered_matches.sort(key=lambda x: x.distance)
                # Select only the best matches (e.g., the first 10 matches)
                good_matches = filtered_matches[:15]

                ##### Find the displacements
                delta_y = 0
                delta_x = 0

                for match in good_matches:
                    new_kp_idx = match.trainIdx
                    ref_kp_idx = match.queryIdx
                    new_keypoint = new_kp[new_kp_idx]
                    ref_keypoint = self.reference_kp[ref_kp_idx]

                    delta_y += new_keypoint.pt[1] - ref_keypoint.pt[1]
                    delta_x += new_keypoint.pt[0] - ref_keypoint.pt[0]
            
                # Uncomment here to just calculate the average displacement
                delta_x /= len(good_matches)
                delta_y /= len(good_matches) 
                time_t = timestamp - self.start_time
                print("displacement for file name {} is delta_x = {}, delta_y = {}, at time {}".format(filename, delta_x * self.pixelSpacing, delta_y * self.pixelSpacing, time_t))


                py = delta_x * self.pixelSpacing
                pz = delta_y * self.pixelSpacing
                
                self.probeArray.add_item([[py, pz, time_t]])
                print("probeArray's length is ", self.probeArray.get_num_element())
                if self.probeArray.is_queue_full():
                    yy, zz, tt = self.probeArray.get_yy_zz_tt()
                    motionModelParams = self.m_motionTracker.fit_sin(tt, zz, yy)
                    json_text_to_parse = str(motionModelParams["amp"]) + " " + str(motionModelParams["period"]) + " " + str(motionModelParams["phase"]) + " " + str(motionModelParams["offset"]) + " " + str(motionModelParams["ny"]) + " " + str(motionModelParams["nz"])
                    print("MotionModelParam is sent ", json_text_to_parse)


                # Draw matched keypoints on the new frame
                self.matchedImage = np.empty((max(reference_image.shape[0], image.shape[0]), reference_image.shape[1]+image.shape[1], 3), dtype=np.uint8)
                cv.drawMatches(refImageWithRoi, self.reference_kp, image, new_kp, good_matches, self.matchedImage)
                images.append(image)
                
                cv.imshow("Matches", self.matchedImage)
                cv.waitKey(33)

        

    ## This function perform presetting of the US-US registration 
    # @param img Referenced Ultrasound Image
    def presettingUSUSWithReferenceImg(self, img):
        return img
        
    #============== END of US-US Preparation =====================

    ## This function process the US-US registration 
    # @param img Realtime Ultrasound Image
    def processUSUS(self, img):
    ############### start to register ###############
        transform_matrix = img
        return transform_matrix
    

    ## This function calculating the bounding box of the kidney roi
    # @param points Points selected
    def bounding_box(self, points):
        x_coord, y_coord = zip(*points)
        return[(min(x_coord), min(y_coord), max(x_coord), max(y_coord))]

    def updateCroppingWindowParams(self, binaryInage):
        m_contours, hierarchy = cv.findContours(binaryInage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_contour = max(self.m_contours, key=cv.contourArea)
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
        return croppedImg
        
    ## Main function of the UGS Client Module for Animal Data
    def client_task(self):
        # motionTracker = MotionTracking() 
        
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
                # if message.device_name == "BinaryImage":
                #     print(message)
                #     #binary_img_data = np.squeeze(message.image.reshape(1,self.imageSizeX,self.imageSizeY,1).transpose(0,1,2))
                #     fn = 'roi_bin.jpg'
                #     save_path 	= './data/kidney_boundary/testOutput'
                #     img_data2d = np.squeeze(message.image.reshape(1,self.imageSizeX,self.imageSizeY).transpose(0,1,2))
                #     img_data = Image.fromarray(img_data2d)
                #     img_data.save(os.path.join(save_path, fn))
 
                #     self.updateCroppingWindowParams(np.asarray(img_data))
                #     img_data = self.cropImage(np.asarray(img_data))
                #     fn = 'roi_bin_cropped.jpg'
                #     img2s = Image.fromarray(img_data)
                #     img2s.save(os.path.join(save_path, fn))

                #     print('new center is at ', self.centerX)
                #     print('saved binary data')
                #     readyToPreset = True
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
                        controurArray = sitk.GetArrayFromImage(self.m_contours) 
                        #process matrix and generate transform
                        image_message = pyigtl.ImageMessage(controurArray, device_name="xContourRef", timestamp=timestamp)
                        transform_message = pyigtl.TransformMessage(self.m_US_matrix, device_name="xMatrixRef", timestamp=timestamp)
                        # Send messagestransform_message
                        client.send_message()
                        client.send_message(image_message)
                    if self.US_US_Registration == 1:
                        log.info('Timestamp: %s - Ultrasound Image Reference Image is received from server!', timestamp_for_log)
                        img = message.image.reshape(self.imageSizeX,self.imageSizeY)
                        print(message.ijk_to_world_matrix)
                        self.probe_pose0 = message.ijk_to_world_matrix
                        
                if message.device_name == 'USImage':
                    
                    log.info('Timestamp: %s - US Image is received from server!', timestamp_for_log)
                    
                    if self.US_US_Registration == 1:
                        img = message.image.reshape(self.imageSizeX,self.imageSizeY)
                        transform_matrix = {1}
                        # Generate transform
                        transform_message = pyigtl.TransformMessage(transform_matrix, device_name="xMatrixUSUS", timestamp=timestamp)
                        # Send messages    
                        client.send_message(transform_message)
                        print(transform_message)

                        # Track the objects below here

                        log.info('Timestamp: %s - Tracking object is done! Transformation Matrix sent back!', timestamp_for_log)
                        py, pz = self.m_motionTracker.preprocessing_data(self.probe_pose0, message.ijk_to_world_matrix, transform_matrix)  
                        self.probeArray.add_item([[py, pz, timestamp]])
                        print("probeArray's length is ", self.probeArray.get_num_element())
                        if self.probeArray.is_queue_full():
                            yy, zz, tt = self.probeArray.get_yy_zz_tt()
                            motionModelParams = self.m_motionTracker.fit_sin(tt, zz, yy)
                            json_text_to_parse = str(motionModelParams["amp"]) + " " + str(motionModelParams["period"]) + " " + str(motionModelParams["phase"]) + " " + str(motionModelParams["offset"]) + " " + str(motionModelParams["ny"]) + " " + str(motionModelParams["nz"])
                            #json_text_to_parse = '{ "amp":"'+str(motionModelParams["amp"])+'", "period":"'+str(motionModelParams["period"])+'", "phase":"'+str(motionModelParams["phase"])+'", "offset":"'+str(motionModelParams["offset"])+'", "ny":"'+str(motionModelParams["ny"])+'", "nz":"'+str(motionModelParams["nz"])+'"}'
                            modelParamMessage = pyigtl.StringMessage(json_text_to_parse, device_name="MotionModelParam", timestamp=timestamp)
                            client.send_message(modelParamMessage, wait=True)
                            print("MotionModelParam is sent ", json_text_to_parse)

                        
                #print(message)
                if message.device_name == 'String':
                    print(message)
                print('\n')   
                
                print('Time to process one loop', timer() - start)
        # Send messages
        client.send_message(string_message)
        
