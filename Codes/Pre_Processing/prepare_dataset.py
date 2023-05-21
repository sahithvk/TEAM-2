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

from utils import is_dir_path,segment_lung
from pylidc.utils import consensus
from PIL import Image

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset','MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_MASK'))
META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))

IMAGE_3D_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_3D_PATH'))
IMAGE_3D_MASK_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_3D_MASK_PATH'))

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR, mask_threshold, padding,IMAGE_3D_DIR,IMAGE_3D_MASK_DIR, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta = pd.DataFrame(index=[],columns=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])
        
        self.img_3d_path = IMAGE_3D_DIR
        self.img_3d_mask_path = IMAGE_3D_MASK_DIR
        self.patient_id=LIDC_Patients_list

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
        tmp = pd.Series(meta_list,index=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])
        self.meta = self.meta.append(tmp,ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)
        
        if not os.path.exists(self.img_3d_path):
            os.makedirs(self.img_3d_path)
        if not os.path.exists(self.img_3d_mask_path):
            os.makedirs(self.img_3d_mask_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)
        
        IMAGE_3D_DIR = Path(self.img_3d_path)
        IMAGE_3D_MASK_DIR = Path(self.img_3d_mask_path)

        for patient in tqdm(self.IDRI_list):
            
            image_3d=[]
            
            patient_image_3d_dir = IMAGE_3D_DIR / patient
            Path(patient_image_3d_dir).mkdir(parents=True, exist_ok=True)
            patient_image_3d_mask_dir = IMAGE_3D_MASK_DIR / patient
            Path(patient_image_3d_mask_dir).mkdir(parents=True, exist_ok=True)
            
            pid = patient #LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            print("scaaaaaaaaaannnn")
            print(scan)
            ###############
            ann = pl.query(pl.Annotation).filter(pl.Annotation.scan_id == scan.id).sort()
            ##############
            nodules_annotation = scan.cluster_annotations()
            print("annotaaaaaaaaaaaaa")
            #print(nodules_annotation[0][1])
            print(nodules_annotation)
            vol = scan.to_volume()
            #########
            #resized_image,resized_mask= ann.uniform_cubic_resample()
            
            print("resisssssssssssssssss")
            print(ann)
            #print(resized_image.shape)
            
            print("HELlo")
            print(vol.shape)
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))
		
            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)
            
            image_3d=vol
            

            if len(nodules_annotation) > 0:
                # Patients with nodules
                
#                 mask_fake, cbbox_fake, masks_fake = consensus(nodules_annotation,self.c_level,self.padding)
#                 print(vol[cbbox_fake].shape)

                image_3d_mask=[[[False for i in range(vol.shape[2])] for j in range(512)] for k in range(512)]
                image_3d_mask=np.array(image_3d_mask)
        
                for nodule_idx, nodule in enumerate(nodules_annotation):
                # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)
                    lung_np_array = vol[cbbox]
                    
                    image_3d_mask[cbbox]=mask
                    
                    
#                     print("maskdotshapee2")
#                     print(image_3d_mask.shape)
#                     print("-----------------")
#                     print(cbbox[2].stop)
#                     print("-----------------")
#                     print(cbbox[2].step)
#                     print("-----------------")
#                     print(lung_np_array)
                    

                    # We calculate the malignancy information
                    malignancy, cancer_label = self.calculate_malignancy(nodule)
                    resized_image,resized_mask= mask.uniform_cubic_resample()

                    for nodule_slice in range(mask.shape[2]):
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        if np.sum(mask[:,:,nodule_slice]) <= self.mask_threshold:
                            continue
                        # Segment Lung part only
                        lung_segmented_np_array = segment_lung(lung_np_array[:,:,nodule_slice])
                        # I am not sure why but some values are stored as -0. <- this may result in datatype error in pytorch training # Not sure
#                         print(type(lung_segmented_np_array))
#                         im_fake = Image.fromarray(lung_segmented_np_array)
#                         im_fake.save("seg_file.jpg")
#                         scipy.misc.toimage(lung_segmented_np_array, cmin=0.0, cmax=...).save('outfile.jpg')
#                         matplotlib.image.imsave('nabongu.png', lung_segmented_np_array)
    
#                         lung_segmented_np_array.save('segmented.jpg')

                        lung_segmented_np_array[lung_segmented_np_array==-0] =0
#                         matplotlib.image.imsave('nabongu_0.png', lung_segmented_np_array)
                        
#                         image_3d.append(lung_segmented_np_array)
                        
                        # This itereates through the slices of a single nodule
                        # Naming of each file: NI= Nodule Image, MA= Mask Original
                        nodule_name = "{}_NI{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        mask_name = "{}_MA{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        meta_list = [pid[-4:],nodule_idx,prefix[nodule_slice],nodule_name,mask_name,malignancy,cancer_label,False]

                        self.save_meta(meta_list)
                        np.save(patient_image_dir / nodule_name,lung_segmented_np_array)
                        np.save(patient_mask_dir / mask_name,mask[:,:,nodule_slice])
            else:
                print("Clean Dataset",pid)
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                #There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                for slice in range(vol.shape[2]):
#                     if slice >50:
#                         break
                    lung_segmented_np_array = segment_lung(vol[:,:,slice])
                    lung_segmented_np_array[lung_segmented_np_array==-0] =0
                    lung_mask = np.zeros_like(lung_segmented_np_array)
                    
#                     image_3d.append(lung_segmented_np_array)

                    #CN= CleanNodule, CM = CleanMask
#                     nodule_name = "{}/{}_CN001_slice{}".format(pid,pid[-4:],prefix[slice])
#                     mask_name = "{}/{}_CM001_slice{}".format(pid,pid[-4:],prefix[slice])
#                     meta_list = [pid[-4:],slice,prefix[slice],nodule_name,mask_name,0,False,True]
#                     self.save_meta(meta_list)
#                     np.save(patient_clean_dir_image / nodule_name, lung_segmented_np_array)
#                     np.save(patient_clean_dir_mask / mask_name, lung_mask)


            image_3d=np.array(resized_image)
            np.save(patient_image_3d_dir / patient,image_3d)
            image_3d_mask=np.array(resized_mask)
            np.save(patient_image_3d_mask_dir / patient,image_3d_mask)
        
        print("Saved Meta data")
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)



if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file 
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()


    test= MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR,mask_threshold,padding,IMAGE_3D_DIR,IMAGE_3D_MASK_DIR,confidence_level)
    test.prepare_dataset()
