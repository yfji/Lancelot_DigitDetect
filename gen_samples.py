# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:25:52 2017

@author: yfji
"""

import os
import cv2
import numpy as np

root=os.getcwd()
pic_dir=os.path.join(root, 'pic')
file_lst_path=os.path.join(root, 'file_lst.txt')

def create_file_lst(dir_, files):
    dir_files=os.listdir(dir_)
    for f in dir_files:
        f=os.path.join(dir_, f)
        if os.path.isdir(f):
            create_file_lst(f, files)
        else:
            files.append(f)

def findDigitArea(image, withPad=True):
    assert(len(image.shape)==2)
    _,image=cv2.threshold(image, 50,255, cv2.THRESH_OTSU) 
    rect=np.where(image==0)
    lefttop=(rect[0].min(),rect[1].min())
    rightbottom=(rect[0].max(),rect[1].max())
    #digit=image[lefttop[0]:rightbottom[0]+1,lefttop[1]:rightbottom[1]+1]
    w=rightbottom[1]-lefttop[1]+1
    h=rightbottom[0]-lefttop[0]+1
    bndbox=[lefttop[1],lefttop[0],w,h]
    if withPad:
        im_w=image.shape[1]
        im_h=image.shape[0]
        max_pad=max(2,min(4,min(w,h)/16))
        pad=np.random.randint(0,max_pad)
        bndbox[0]==max(0,lefttop[1]-pad)
        bndbox[1]=max(0,lefttop[0]-pad)
        bndbox[2]=min(im_w-bndbox[1]+1,w+2*pad)
        bndbox[3]=min(im_h-bndbox[0]+1,h+2*pad)
    return bndbox        
    
def cropSamples(dir_, file_lst):
    f=open(file_lst, 'r')
    samples=f.readlines()
    for sample in samples:
        if len(sample)<3:
            continue
        items=sample.rstrip().split('/')
        file_name=items[-1]
        file_name_no_ext=file_name[:file_name.rfind('.')]
        ext=file_name[file_name.rfind('.'):]
        concat_name=file_name_no_ext+'_'+items[-2]+'_'+items[-3]+ext
        print(concat_name)
        image=cv2.imread(sample.rstrip(),0)
        bndbox=map(int,findDigitArea(image, withPad=True))
        roi=image[bndbox[1]:bndbox[1]+bndbox[3],bndbox[0]:bndbox[0]+bndbox[2]]
        cv2.imwrite(os.path.join(dir_,concat_name), roi)
    print('\n'+'finish')
        
    
if __name__=='__main__':
    cropSamples('./pic/crop','./pos_samples.txt')
#    files=[]
#    create_file_lst(pic_dir, files)
#    with open(file_lst_path, 'w') as f:
#        for line in files:
#            f.write(line+'\n')
