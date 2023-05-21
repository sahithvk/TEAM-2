# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high
import matplotlib
import cv2

from utils import *
from pylidc.utils import consensus
from PIL import Image

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))

IMAGE_3D_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_3D_PATH'))
IMAGE_3D_MASK_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_3D_MASK_PATH'))
META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))

IMAGE_3D_LUNG_SEG_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_3D_LUNG_SEG'))
IMAGE_3D_LUNG_SEG_MASK_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_3D_LUNG_SEG_MASK'))

IMAGE_3D_NORM_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_3D_NORM'))
IMAGE_3D_NORM_MASK_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_3D_NORM_MASK'))


#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, mask_threshold, padding,META_DIR,IMAGE_3D_DIR,IMAGE_3D_MASK_DIR, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta_path = META_DIR
        self.meta = pd.DataFrame(index=[],columns=['patient_id','nodule_no','is_cancer'])
        
        self.img_3d_path = IMAGE_3D_DIR
        self.img_3d_mask_path = IMAGE_3D_MASK_DIR

        self.img_3d_lung_seg = IMAGE_3D_LUNG_SEG_DIR
        self.img_3d_lung_seg_mask = IMAGE_3D_LUNG_SEG_MASK_DIR

        self.img_3d_norm = IMAGE_3D_NORM_DIR
        self.img_3d_norm_mask = IMAGE_3D_NORM_MASK_DIR
        
    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if  malignancy > 3:
            return malignancy,True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'
    def save_meta(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','nodule_no','is_cancer'])
        self.meta = self.meta.append(tmp,ignore_index=True)

    def prepare_dataset(self):

        try:
            if not os.path.exists(self.img_3d_path):
                os.makedirs(self.img_3d_path)
            if not os.path.exists(self.img_3d_mask_path):
                os.makedirs(self.img_3d_mask_path)
            if not os.path.exists(self.meta_path):
                os.makedirs(self.meta_path)
                
            if not os.path.exists(self.img_3d_lung_seg):
                os.makedirs(self.img_3d_lung_seg_mask)
            if not os.path.exists(self.meta_path):
                os.makedirs(self.meta_path)

            IMAGE_3D_DIR = Path(self.img_3d_path)
            IMAGE_3D_MASK_DIR = Path(self.img_3d_mask_path)
            
            IMAGE_3D_LUNG_SEG_DIR = Path(self.img_3d_lung_seg)
            IMAGE_3D_LUNG_SEG_MASK_DIR = Path(self.img_3d_lung_seg_mask)
            
            IMAGE_3D_NORM_DIR = Path(self.img_3d_norm)
            IMAGE_3D_NORM_MASK_DIR = Path(self.img_3d_norm_mask)

            for patient in tqdm(self.IDRI_list):


                pid = patient #LIDC-IDRI-0001~
                scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
                print("spacing ",scan.slice_thickness,scan.slice_zvals.shape[0]) 

                if( (scan.slice_thickness > 2.5)  ) :
                    meta_list = [pid[-4:],"thickness > 2.5","error"]
                    self.save_meta(meta_list)
                    continue
                if((scan.slice_zvals.shape[0] < 100)):
                    meta_list = [pid[-4:],"zval < 100","error"]
                    self.save_meta(meta_list)
                    continue

                nodules_annotation = scan.cluster_annotations()
                print("annotaaaaaaaaaaaaa")
                print(nodules_annotation)
                vol = scan.to_volume()

                print("HELlo")
                print(vol.shape)
                print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))

                if(vol.shape[2]<100 ):
                    meta_list = [pid[-4:],"vol shape < 100","error"]
                    self.save_meta(meta_list)
                    continue
                if( scan.slice_zvals.shape[0]!=vol.shape[2]):
                    meta_list = [pid[-4:],"z val != shape","error"]
                    self.save_meta(meta_list)
                    continue

                ##ikkada rayali seg lung

                patient_image_3d_dir = IMAGE_3D_DIR / patient
                Path(patient_image_3d_dir).mkdir(parents=True, exist_ok=True)
                patient_image_3d_mask_dir = IMAGE_3D_MASK_DIR / patient
                Path(patient_image_3d_mask_dir).mkdir(parents=True, exist_ok=True)
                
                patient_image_3d_lung_seg_dir = IMAGE_3D_LUNG_SEG_DIR / patient
                Path(patient_image_3d_lung_seg_dir).mkdir(parents=True, exist_ok=True)
                patient_image_3d_lung_seg_mask_dir = IMAGE_3D_LUNG_SEG_MASK_DIR / patient
                Path(patient_image_3d_lung_seg_mask_dir).mkdir(parents=True, exist_ok=True)
                
                patient_image_3d_norm_dir = IMAGE_3D_NORM_DIR / patient
                Path(patient_image_3d_norm_dir).mkdir(parents=True, exist_ok=True)
                patient_image_3d_norm_mask_dir = IMAGE_3D_NORM_MASK_DIR / patient
                Path(patient_image_3d_norm_mask_dir).mkdir(parents=True, exist_ok=True)
                    
                
                image_3d=vol
#                 np.save(patient_image_3d_dir / patient,image_3d)
                image_3d_lung_seg=[]
    
                for seg in image_3d.T:
                    image_3d_lung_seg.append( (segment_lung(seg.T)).T )
                
                image_3d_lung_seg=np.array(image_3d_lung_seg).T
                
                var1=segment_lung_mask(image_3d,False)
                var2=segment_lung_mask(image_3d,True)
                var2=var2-var1
                image_3d=var2*image_3d
#                 print(image_3d)
                image_3d_norm=normalise(image_3d)
#                 print(image_3d_norm)

                image_3d_mask=[[[False for i in range(vol.shape[2])] for j in range(512)] for k in range(512)]
                image_3d_mask=np.array(image_3d_mask)

                if len(nodules_annotation) > 0:
                    # Patients with nodules

    #                 mask_fake, cbbox_fake, masks_fake = consensus(nodules_annotation,self.c_level,self.padding)
    #                 print(vol[cbbox_fake].shape)

                    summ=0
                    coun=0

                    for nodule_idx, nodule in enumerate(nodules_annotation):
                    # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                    # This current for loop iterates over total number of nodules in a single patient
                        mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)
                        lung_np_array = vol[cbbox]


    #                     print("maskshape:")
    #                     print(mask.shape)
    #                     print(image_3d_mask.shape)


    #                     print("maskdotshapee2")
    #                     print(image_3d_mask.shape)
    #                     print("-----------------")
                        print("cbbox",cbbox)
    #                     print("-----------------")
    #                     print(cbbox[2].step)
    #                     print("-----------------")
    #                     print(lung_np_array)


                        # We calculate the malignancy information
                        malignancy, cancer_label = self.calculate_malignancy(nodule)
                        if malignancy < 3 :
                            continue

                        summ+=((cbbox[2].start+cbbox[2].stop)//2)
                        coun+=1
                        
                        print("image_3d_mask's shape",image_3d_mask.shape," || ",mask.shape)
                        print("image of cbbox",image_3d_mask[cbbox].shape )
                        
                        image_3d_mask[cbbox]=mask

                    print("before count if: and cancer nodules")
                    print(summ)
                    print(coun)
                    if coun==0:
                        summ=vol.shape[2]//2
                    else :
                        summ=summ//coun
                    print("after count if:")
                    print(summ)
                    print(coun)

                    if summ<50 :
                        summ=50
                    elif summ > (vol.shape[2]-50) :
                        summ=vol.shape[2]-50
                    print("final in if")
                    print(summ)
                    print(coun)

                else:
                    summ=vol.shape[2]//2


                image_3d=np.array(image_3d)
                image_3d=image_3d.T[summ-50:summ+50]
                image_3d=image_3d.T
                image_3d=rsize_image(image_3d)
                
                image_3d_norm=image_3d_norm.T[summ-50:summ+50]####_norm_###
                image_3d_norm=image_3d_norm.T####_norm_###
                image_3d_norm=rsize_image_norm(image_3d_norm)####_norm_###
                
                image_3d_lung_seg=image_3d_lung_seg.T[summ-50:summ+50]###_seg_##
                image_3d_lung_seg=image_3d_lung_seg.T###_seg_###
                image_3d_lung_seg=rsize_image_norm(image_3d_lung_seg)###_seg_##
                
                image_3d_mask=np.array(image_3d_mask,dtype='uint8')
                image_3d_mask=image_3d_mask.T[summ-50:summ+50]
                image_3d_mask=image_3d_mask.T
                
                image_3d_norm_mask = (image_3d_norm)*(rsize_image_norm(image_3d_mask))####_norm_###
                
                image_3d_lung_seg_mask=(image_3d_lung_seg)*(rsize_image_norm(image_3d_mask))###_seg_###
                
                image_3d_mask=image_3d*(rsize_image(image_3d_mask))####_whithout_norm_###
                
                np.save(patient_image_3d_dir / patient,image_3d)
                np.save(patient_image_3d_mask_dir / patient,image_3d_mask)
                ####_norm_###
                np.save(patient_image_3d_norm_dir / patient,image_3d_norm)
                np.save(patient_image_3d_norm_mask_dir / patient,image_3d_norm_mask)
                ####_Seg_###
                np.save(patient_image_3d_lung_seg_dir / patient,image_3d_lung_seg)
                np.save(patient_image_3d_lung_seg_mask_dir / patient,image_3d_lung_seg_mask)

                meta_list = [pid[-4:],coun,bool(coun)]
                self.save_meta(meta_list)

            print("Saved Meta data")
            self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)
        
        except:
            print("some error ocurred at",pid,"Saved Meta data")
            self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)
            

if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file 
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()
    print(len(LIDC_IDRI_list))

    test= MakeDataSet(LIDC_IDRI_list,mask_threshold,padding,META_DIR,IMAGE_3D_DIR,IMAGE_3D_MASK_DIR,confidence_level)
    test.prepare_dataset()