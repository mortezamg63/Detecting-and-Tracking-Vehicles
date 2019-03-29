import cv2
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import utils
import KalmanFilter as kf_tracker
from settings import *

tracker_list =[] # list for trackers


def Get_tracker_list():
    return tracker_list

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = utils.box_iou2(trk,det) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       

def pipline(frame, roi_boxes):
        global tracker_list
        global track_id_list
        z_box = [utils.correct_position(x) for x in roi_boxes]
        x_box = []

        if len(tracker_list) > 0:
            for trk in tracker_list:
                x_box.append(trk.box)

        matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)  

        # Deal with matched detections     
        if matched.size >0:
            for trk_idx, det_idx in matched:
                z = z_box[det_idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk= tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                x_box[trk_idx] = xx
                tmp_trk.box =xx
                tmp_trk.hits += 1
                tmp_trk.no_losses = 0
        
        # Deal with unmatched detections      
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = kf_tracker.Tracker() # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.id = track_id_list#.popleft() # assign an ID for the tracker
                track_id_list+=1
                tracker_list.append(tmp_trk)
                x_box.append(xx)
        
        # Deal with unmatched tracks       
        if len(unmatched_trks)>0:
            for trk_idx in unmatched_trks:
                tmp_trk = tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box =xx
                x_box[trk_idx] = xx
                                    
        # Book keeping
        # deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)          
        
        tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    
        return tracker_list
     
