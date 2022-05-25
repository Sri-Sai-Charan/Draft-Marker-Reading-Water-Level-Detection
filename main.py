from pickletools import float8
import sys
sys.path.append('core')

import argparse
import os
import cv2 as cv
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'




def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().detach().numpy()
    flo = flo[0].permute(1,2,0).cpu().detach().numpy()
    
    flo = flow_viz.flow_to_image(flo)


    return flo[:, :, [2,1,0]]/255.0

def load_image(image):
    img = image.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def main(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    vid_cap = cv.VideoCapture("input/water_level.mp4")

    _, prev_frame = vid_cap.read()    
    prev_frame = load_image(prev_frame)

    cap = cv.VideoCapture("input/water_level.mp4")

    # Setting Up video writer
    (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = cap.get(cv.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cap.get(cv.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    
    vid = cv.VideoWriter('./Water_level_detection.avi',cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    while True:
        success, current_frame = vid_cap.read()

        if not success:
            break
        temp_img = current_frame.copy()
        crop = np.zeros_like(current_frame)
        crop[:,180:350] = 255
        current_frame = cv.bitwise_and(crop,current_frame)


        

        current_frame = load_image(current_frame)        

        padder = InputPadder(current_frame.shape)
        image1, image2 = padder.pad(current_frame, prev_frame)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        result = viz(image1, flow_up)

        prev_frame = current_frame
        result = result*255
        heat_map_img = result.astype(np.uint8)
        
        orginal_gray = cv.cvtColor(temp_img, cv.COLOR_BGR2GRAY)
        gray_img = cv.cvtColor(heat_map_img, cv.COLOR_BGR2GRAY)
        ret,thresh_img = cv.threshold(gray_img , 230 ,255,cv.THRESH_BINARY)

        masked_img = cv.bitwise_and(orginal_gray,thresh_img)
        crop = np.zeros_like(masked_img)
        crop[:,180:350] = 255
        masked_img = cv.bitwise_and(crop,masked_img)


        _ , masked_thresh = cv.threshold(masked_img , 230 ,255,cv.THRESH_BINARY)
        kernel = np.ones((7,7),np.uint8)


        masked_thresh_gradient = cv.morphologyEx(masked_thresh, cv.MORPH_GRADIENT, kernel)
        masked_thresh_gradient_canny = cv.Canny(masked_thresh_gradient,100,200,2)


        contours, hierarchy  = cv.findContours(masked_thresh_gradient_canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        blank_testing = np.zeros_like(temp_img)
        avg_arc_length = []
        canvas = np.zeros_like(temp_img)
        for i in contours:
            
            if 300 > cv.arcLength(i,False)>250:
                cv.drawContours(blank_testing, [i], -1, (0,255,0), 3)
                avg_arc_length.append(cv.arcLength(i,False))
 

                rect = cv.minAreaRect(i)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(temp_img,[box],0,(0,0,255),2)
                cv.drawContours(canvas,[box],0,(0,0,255),2)
                warp_image(box,masked_thresh)


            if 250> cv.arcLength(i,False) > 110:
                cv.drawContours(blank_testing, [i], -1, (255,255,255), 3)
                avg_arc_length.append(cv.arcLength(i,False))

                
                rect = cv.minAreaRect(i)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(temp_img,[box],0,(0,0,255),2)
                cv.drawContours(canvas,[box],0,(0,0,255),2)
                warp_image(box,masked_thresh)
        index_ = np.argwhere(canvas[:,:,2]==255)
        try:
            index_max = np.max(index_[:,0])
            cv.imshow("Canvas",canvas)
            no_frames_flag = False
        except:
            no_frames_flag = True
        water_level = ""

        # Determine Water-level based on pixel value
        if index_max>25:
            water_level="10.2 M"
        if index_max>50:
            water_level="10 M"
        if index_max>70:
            water_level="9.8 M"
        if index_max>90:
            water_level="9.6 M"
        if index_max>110:
            water_level="9.4 M"
        if index_max>140:
            water_level="9.2 M"
        if index_max>160:
            water_level="9 M"
        if index_max>190:
            water_level="8.8 M"
        if index_max>220:
            water_level="8.6 M"
        if index_max>240:
            water_level="8.4 M"
        if index_max>270:
            water_level="8.2 M"

        if not no_frames_flag:
            cv.putText(temp_img,"Water level : "+water_level, (30,30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
        else:
            cv.putText(temp_img, "Not Detected", (30,30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)
            

        cv.namedWindow("Masked_thresh",cv.WINDOW_NORMAL)
        cv.imshow("Masked_thresh",masked_thresh)

        cv.imshow("Water Level Detection",temp_img)
        vid.write(temp_img)
        if cv.waitKey(1) == ord('q'):
            break

    vid_cap.release()
    cv.destroyAllWindows()



def warp_image(input_pts,frame):
    inputs = np.float32(input_pts)

    x_arr = []
    y_arr  = []

    for i in inputs:
        x,y = i.ravel()
        x_arr.append(x)
        y_arr.append(y)
    
    x_max = np.max(x_arr[:])
    y_max = np.max(y_arr[:])
    x_min = np.min(x_arr[:])
    y_min = np.min(y_arr[:])

    my_corners =[]
    my_corners.append([x_min,y_min]) # top left    
    my_corners.append([x_max,y_min]) # top right
    my_corners.append([x_max,y_max]) # bottom right
    my_corners.append([x_min,y_max]) # Bottom left

    my_corners = np.float32(my_corners)
    output_pts = np.array([[0,0],
                    [200,0],
                    [200,100],
                    [0,100]],dtype=np.float32)
    M = cv.getPerspectiveTransform(my_corners,output_pts)
    out = cv.warpPerspective(frame,M,(200, 100),flags=cv.INTER_LINEAR)
    cv.imshow("warped_image",out)

    # CNN - Implementation
    # from keras.callbacks import ModelCheckpoint, EarlyStopping
    # checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, 
    #                             save_weights_only=False, mode='auto', save_freq=1)
    # early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
    # hist = model.fit(traindata,steps_per_epoch=377//32,validation_data= testdata,
    #                         validation_steps=206//32,epochs=10,callbacks=[checkpoint,early])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    main(args)


