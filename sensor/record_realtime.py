import os,sys,inspect
currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
basedir=currentdir+'/../'
sys.path.append(basedir+'sensor/')

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import pyrealsense2 as rs
import sys

from scipy import stats, ndimage
from Realsense import Realsense


if __name__=="__main__":
    '''user setting'''
    save_enabled = False
    frame_saved = 0
    frame = 0
    save_filepath = './save_pth/'
    
    ##--init realsense--##
    realsense = Realsense.Realsense()
     
    try:
        while True:
            if frame is 0:
                print('start recording..')
                
            realsense.run()

            depth=realsense.getDepthImage()            
            color=realsense.getColorImage()
            
            depth=depth.astype(np.uint16)
            
            depth_seg=depth.copy() 
            depth_seg[depth_seg>500]=0

            t = depth_seg / np.max(depth_seg)

            #cv2.imshow('depth',np.uint8(depth))
            cv2.imshow('depth_seg',np.uint8(depth_seg))
            cv2.imshow('depth_seg 2', t)
            cv2.imshow('color',color)
            
            #save original image
            if save_enabled is True and frame>-1:
                cv2.imwrite(save_filepath+'depth_%d.png'%frame_saved,depth)
                cv2.imwrite(save_filepath+'color_%d.png'%frame_saved,color)
                frame_saved+=1

            cv2.waitKey(1)
            frame += 1
            if frame % 100 is 0:
                print('frame..',frame)
            
    finally:
        print('stop device')
        realsense.release()    
        
    print(frame)
        
    
    