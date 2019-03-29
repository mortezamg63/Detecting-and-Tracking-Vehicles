import cv2
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def detection(img, net):
    
    width = img.shape[1]
    height = img.shape[0]

    scale = 0.00392          

    # create input blob
    blob = cv2.dnn.blobFromImage(img, scale,  (416, 416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    #initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.5:
                center_x = int(detection[0]*width)                
                center_y = int(detection[1]*height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y -h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x,y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, confidences, class_ids, indices


def draw_box_label(img, bbox_cv2, label, box_color=(0, 255, 0), show_label=False):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    # left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]
    left, top, right, bottom = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
    
    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 1)
    cv2.putText(img, str(label), (left-2, top-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
    
    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left-2, top-45), (right+2, top), box_color, -1, 1)
        
        # Output the labels that show the x and y coordinates of the bounding box center.
        text_x= 'x='+str((left+right)/2)
        cv2.putText(img,text_x,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)
        text_y= 'y='+str((top+bottom)/2)
        cv2.putText(img,text_y,(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)
    
    return img    



def box_iou2(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''
    
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
  
    return float(s_intsec)/(abs(s_a) + abs(s_b) -s_intsec)

def correct_position(box):    
    x = int(round(abs(box[0])))
    y = int(round(abs(box[1])))
    w = int(round(abs(box[2])))
    h = int(round(abs(box[3])))
    startX, startY, endX, endY = x,y, x+w, y+h
    return [startX, startY, endX, endY]

def draw_bounding_box_KalmanFilter(img, x, y, x_plus_w, y_plus_h, label, color=(0,255,0)):#class_id, confidence, box):
    x, y, x_plus_w, y_plus_h = int(x), int(y), int(x_plus_w), int(y_plus_h)
    label = str(label)

    # color = (0,255,0)
    # cv2.imwrite('extracted_objects/')

    cv2.rectangle(img, (x,y), (x_plus_w, y_plus_h), color, 1)
    cv2.putText(img, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
