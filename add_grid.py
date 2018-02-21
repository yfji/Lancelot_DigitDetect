# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:11:43 2017

@author: yfji
"""

import cv2
import os
import numpy as np

def addGrid(image):
    gImage=None
    if len(image.shape)==3:
        gImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    else:
        gImage=image.copy()
    _,biMap=cv2.threshold(gImage, 50, 255, cv2.THRESH_OTSU)
    area=np.where(biMap==0)
    canvas=np.ones((700,600))*255
    std_w=40
    max_shift_w=10
    w=np.random.randint(-max_shift_w, max_shift_w+1)+std_w
    cv2.rectangle(canvas, (100,100),(500,600),0,-1)
    cv2.rectangle(canvas, (100+w,100+w),(500-w,600-w),255,-1)
    diff_x=int(0.5*canvas.shape[1]-0.5*(area[1].max()+area[1].min()))
    diff_y=int(0.5*canvas.shape[0]-0.5*(area[0].max()+area[0].min()))
    shift_y=area[0]+diff_y
    shift_x=area[1]+diff_x
    canvas[shift_y,shift_x]=0
    
    std_line_w=1
    max_line_shift_w=1
    line_tick_x=np.linspace(100+w,450,7).astype(np.int32)
    line_tick_y=np.linspace(100+w,550,9).astype(np.int32)
    for i in range(1,len(line_tick_x)-1):
        line_w=np.random.randint(-max_line_shift_w,max_line_shift_w+1)+std_line_w
        if line_w>0:
            cv2.line(canvas, (line_tick_x[i],100+w),(line_tick_x[i],600-w), 0, line_w) 
    for i in range(1,len(line_tick_y)-1):
        line_w=np.random.randint(-max_line_shift_w,max_line_shift_w+1)+std_line_w
        if line_w>0:
            cv2.line(canvas, (100+w, line_tick_y[i]),(500-w,line_tick_y[i]), 0, line_w)
    return canvas

def addGridForAllSamples():
    file_lst='/home/jyf/Workspace/C++/UAV2017/file_lst.txt'
    target_lst='//home/jyf/Workspace/C++/UAV2017/grid_thin.txt'
    grid_root=os.path.join(os.getcwd(),'pic/grids/thin_lines')
    f=open(file_lst,'r')
    tf=open(target_lst,'w')
    image_names=f.readlines()
    for image_name in image_names:
        image_name=image_name.rstrip()
        dir_names=image_name.split('/')
        name=dir_names[-1]
        if name[-4:]=='.bmp':
            continue
        sub_dir=dir_names[-2]
        if not os.path.exists(os.path.join(grid_root, sub_dir)):
            os.mkdir(os.path.join(grid_root, sub_dir))
        canvas=addGrid(cv2.imread(image_name))
        grid_name=os.path.join(grid_root, sub_dir, name)
        tf.write(grid_name+'\n')
        cv2.imwrite(grid_name, canvas)
    f.close()
    tf.close()

if __name__=='__main__':
    image_path='/home/jyf/Workspace/C++/UAV2017/pic/fonts/5_xingkai.jpg'
    addGridForAllSamples()
#    canvas=addGrid(cv2.imread(image_path))
#    cv2.imshow('align', canvas)
#    cv2.waitKey()
#    cv2.destroyAllWindows()
