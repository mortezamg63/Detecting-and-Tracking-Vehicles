import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import KalmanFilter as kf_tracker
from settings import *
import multitracking
import utils

def main():
    # loading pre-trained model and config file    
    net = cv2.dnn.readNet(wieght_arg, config_arg)
    print("Graph is loaded ")
    capture = cv2.VideoCapture(input_video)    
    ret, frame = capture.read()
    h,w,c = frame.shape 


    plt.ion()
    index = 0

    # Is there any frame to read?
    while ret:
        index += 1
        print(index)
        ret, frame = capture.read() 
        frame = cv2.resize(frame, (640,320), interpolation = cv2.INTER_LINEAR) 
        
        # feeding tensor to loaded model and obtaining the bounding boxes of detected objects
        # roi_boxes, roi_confidences, roi_class, roi_indices = utils.detection(frame, net)
        roi_boxes, _, _, _ = utils.detection(frame, net)

        tracker_list = multitracking.pipline(frame, roi_boxes)

        # The list of tracks to be annotated  
        good_tracker_list =[]
        for trk in tracker_list:
            if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
                good_tracker_list.append(trk)
                x_cv2 = trk.box
                if debug:
                    print('updated box: ', x_cv2)
                    print()
                frame= utils.draw_box_label(frame, x_cv2, trk.id) # Draw the bounding boxes on the  images
                                                # draw line here
        
        plt.imshow(frame)
        plt.pause(0.002)
   

if __name__ == "__main__":
    main()