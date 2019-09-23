import sys 
sys.path.insert(0,'/home/fapsros/anaconda3/lib/python3.7/site-packages') 
import cv2
import numpy as np 
import glob
import ruamel_yaml as yaml 
import io
import os
import shutil 
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from PIL import Image
#import matplotlib.pyplot as plt

interval = 31 # 40, 17, 31



color_label_directory = 'colorlabel'
parent_dir = './images'
path = os.path.join(parent_dir, color_label_directory)
if not os.path.exists(path):
        os.mkdir(path)
files = [file for file in glob.glob('./images/*_color_labels.png')]

for f in files:
        shutil.move(f,path)


Label_Images = [cv2.imread(file,1) for file in sorted(glob.glob('./images/*_labels.png'))]        # 1->linemod, 0->ycb for segmentation
Color_Images = [cv2.imread(file,1) for file in sorted(glob.glob('./images/*_rgb.png'))]          # order is not guranteed with glob.glob() only
Depth_Images = [Image.open(file) for file in sorted(glob.glob('./images/*_depth.png'))]





list_label = os.listdir('./segmentation/mask/')
number_label_file = len(list_label)

list_color = os.listdir('./segmentation/rgb/')
number_color_file = len(list_color)



list_depth = os.listdir('./segmentation/depth/')
number_depth_file = len(list_depth)



r = [] # store rotation matrices
bb = [] # store bounding box 
t = []  #store translation matrix


Images = []
for filename in sorted(glob.glob('./images/*_labels.png')):
    im = cv2.imread(filename, 0)
    Images.append(im)

for img in Images:
    #img = np.array(img)
    #mask = img ==0
    #img[mask] = 255
    #img[np.logical_not(mask)] = 0
    ret, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    bb.append([x,y,w,h])
    #cv2.rectangle(img, (x,y), (x+w, y+h), (255),2)
    #cv2.imshow("contours", img)

binary = []
for img in Images:
        img = np.array(img)
        mask = img==0
        img[mask] = 0
        img[np.logical_not(mask)] = 255
        binary.append(img)
  
for metafilename in sorted(glob.glob('./images/*_poses.yaml')):
    meta_file = open(metafilename)
    meta = yaml.safe_load(meta_file)
    t_ = [ i *1000 for i in meta['schaltgabel']['pose'][0]]
    t.append(t_)
    quaternion_list = meta['schaltgabel']['pose'][1]
    modified_quaternion = [ quaternion_list[1], quaternion_list[2], quaternion_list[3], quaternion_list[0]]
    r.append(R.from_quat(modified_quaternion).as_dcm().flatten().tolist())



#different notation of quaternions between sicpy and labelfusion
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
#https://github.com/RobotLocomotion/LabelFusion/issues/41


posedata_file = open('./posedata.yml')
posedata = yaml.safe_load(posedata_file)

already = 0

if posedata is None:
        data = {}
else:
        data = posedata
        already = len(posedata)


for idx, item in enumerate(bb):
    if idx % interval ==0:
        data.update(   {int(idx/interval)+already:    [       
                                                        {  'cam_R_m2c': r[idx],
                                                           'cam_t_m2c': t[idx], 
                                                           'obj_bb':    bb[idx], 
                                                           'obj_id':    2  
                                                        }        
                                                ]            
                        }
                    )


with io.open('posedata.yml', 'w') as outfile:
    yaml.dump(data, outfile)








for idx, item in enumerate(binary):
    if idx % interval == 0:
        #plt.imsave('./segmentation/mask/{0}_mask.png'.format(int(idx/interval)+number_label_file), item)
        cv2.imwrite('./segmentation/mask/{:04d}.png'.format(int(idx/interval)+number_label_file), item)

for idx, item in enumerate(Color_Images):
    if idx % interval ==0:
        #plt.imsave('./segmentation/rgb/{0}_rgb.png'.format(int(idx/interval)+number_color_file), item)
        cv2.imwrite('./segmentation/rgb/{:04d}.png'.format(int(idx/interval)+number_color_file), item)



for idx, item in enumerate(Depth_Images):
    if idx % interval ==0:
        item.save('./segmentation/depth/{:04d}.png'.format(int(idx/interval)+number_depth_file))
        #cv2.imwrite('./segmentation/depth/{0}_depth.png'.format(int(idx/interval)+number_depth_file), item.astype(np.uint16))  # keep 16bit same as linemod daatset
        
