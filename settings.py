import numpy as np
input_video = 'video.avi'#'highway.mov'#'highway.mov'##'Tokyo.mp4'#
image_address = 'dog.jpg'
classes_file_address = 'yolov3.txt'
wieght_arg = 'yolov3.weights'
config_arg = 'yolov3.cfg'

# read class names from text
classes = None
with open(classes_file_address, 'r') as f:
    classes = [line.strip() for line in f.readlines()]  


# generate different colors for bounding boxes of diffrent classes
COLORS = np.random.uniform(0,255, size=(len(classes), 3))



# ----------------------multitracking setting----------------------
# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

max_age = 4  # no.of consecutive unmatched detection before 
             # a track is deleted

min_hits =1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= 0 
debug = False