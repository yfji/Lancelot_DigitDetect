	# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 22:04:53 2017

@author: yfji
"""

import os
import numpy as np

root=os.getcwd()
image_dir=os.path.join(root,'/home/jyf/Workspace/C++/UAV2017/pic/crop_no_box')
image_files=os.listdir(image_dir)

def create_file_lst():
	file_lst='pos_samples_crop_no_box.txt'
	f=open(file_lst,'w')

	for image_file in image_files:
	    file_name=os.path.join(image_dir,image_file)
	    f.write(file_name+'\n')
	f.close()

def create_file_lst_random(N=1200):
	file_lst='neg_samples_dataset.txt'
	f=open(file_lst,'w')
	inds=np.random.choice(np.arange(len(image_files)), N, replace=False)

	for ind in inds:
		file_name=os.path.join(image_dir, image_files[ind])
		f.write(file_name+'\n')
	f.close()

if __name__=='__main__':
	create_file_lst()
