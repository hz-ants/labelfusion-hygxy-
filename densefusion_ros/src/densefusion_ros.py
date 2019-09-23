#!/usr/bin/env python
'''
This ros node subscribes to two camera topics: '/camera/color/image_raw' and 
'/camera/aligned_depth_to_color/image_raw' in a synchronized way. It then runs 
semantic segmentation and pose estimation with trained models using DenseFusion
(https://github.com/j96w/DenseFusion). The whole code structure is adapted from: 
(http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber)
'''

import sys
sys.path.insert(0,'/home/fapsros/anaconda3/lib/python3.7/site-packages') # add this line if you encounter "undefined symbol: PyCObject_Type" trigged by import cv2
import cv2                                                               # you may also get "name 'reduce' is not defined" after adding the above line. To resolve this,
import time                                                              # add "from functools import reduce" in file /opt/ros/kinectic/lib/python2.7/dist-packages/
import rospy                                                             # message_filters/__init__.py
import copy
import argparse                                                             
import numpy as np
import numpy.ma as ma 
import message_filters
from sensor_msgs.msg import Image

import time 
import torch
#from segnet import SegNet as segnet                #uncomment if we are using pruned version of segnet
from segnet_original import SegNet as segnet      #uncomment if we are using original version of segnet
import torchvision
from torchvision import transforms
from torch.autograd import Variable

from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor


num_objects = 3
objlist =[1,2,3]
num_points = 500
iteration = 2
bs = 1

knn = KNearestNeighbor(1)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help='name of metal objects to be detected: 0.flansch 1.schaltgabel 2.stift')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_bbox(bbox):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax

DEBUG = False

class pose_estimation:
    def __init__(self, model_, estimator_, refiner_, object_index_, scaled_):
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub],15) # choose the same value as publish rate in rs_camera.launch
        ts.registerCallback(self.callback)                                       # (/home/fapsros/hy_ws/src/realsense-ros/realsense2_camera/launch)
        self.model = model_
        self.estimator = estimator_
        self.refiner = refiner_
        self.object_index = object_index_
        self.scaled = scaled_

        self.cam_cx = 315.2859903333336 
        self.cam_cy = 244.88168334960938
        self.cam_fx = 616.0936279296875
        self.cam_fy = 616.29443359375

        self.xmap = np.array([[j for i in range(640)] for j in range(480)]) 
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        #synchronized rgb and depth messages http://wiki.ros.org/message_filters
        #rospy.Subscriber('/camera/color/image_raw', Image, callback)
        #rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, callback)

        if DEBUG:
            print('subscribed to rgb and depth topic in a sychronized way')

    def callback(self, rgb, depth):
        if DEBUG:
            print ('received depth image of type: ' +depth.encoding)
            print ('received rgb image of type: ' + rgb.encoding)
        #https://answers.ros.org/question/64318/how-do-i-convert-an-ros-image-into-a-numpy-array/
        depth = np.frombuffer(depth.data, dtype=np.uint16).reshape(depth.height, depth.width, -1)
        rgb = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, -1)
        rgb_original = rgb
        #cv2.imshow('depth', depth)

        #time1 = time.time()
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
        rgb = Variable(rgb).cuda()
        semantic = self.model(rgb.unsqueeze(0))
        _, pred = torch.max(semantic, dim=1)
        pred = pred *255
        pred = np.transpose(pred, (1, 2, 0))  # (CxHxW)->(HxWxC)
        #print(pred.shape)

        #ret, threshold = cv2.threshold(pred.cpu().numpy(), 1, 255, cv2.THRESH_BINARY)    #pred is already binary, therefore, this line is unnecessary 
        contours, hierarchy = cv2.findContours(np.uint8(pred),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        rmin, rmax, cmin, cmax = get_bbox([x,y,w,h ])
        #cv2.rectangle(rgb_original,(cmin,rmin), (cmax,rmax) , (0,255,0),2)
        #cv2.imwrite('depth.png', depth)          #save depth image

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth,0))
        mask_label = ma.getmaskarray(ma.masked_equal(pred, np.array(255)))
        mask = mask_depth * mask_label

        #print(rgb.shape)             #torch.Size([3, 480, 640])
        #print(rgb_original.shape)    #(480, 640, 3)
        img = np.transpose(rgb_original, (2, 0, 1))
        img_masked = img[:, rmin:rmax, cmin:cmax]

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        #print("length of choose is :{0}".format(len(choose))) 
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)
        
        if len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:num_points] = 1  # if number of object pixels are bigger than 500, we select just 500
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]  # now len(choose) = 500
        else:
            choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        choose = np.array([choose])

        pt2 = depth_masked
        #print(pt2)
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud /1000

        points = torch.from_numpy(cloud.astype(np.float32))
        choose = torch.LongTensor(choose.astype(np.int32))
        img = norm(torch.from_numpy(img_masked.astype(np.float32)))
        idx = torch.LongTensor([self.object_index])

        img = Variable(img).cuda().unsqueeze(0)
        points = Variable(points).cuda().unsqueeze(0)
        choose = Variable(choose).cuda().unsqueeze(0)
        idx = Variable(idx).cuda().unsqueeze(0)
 
        pred_r, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

        
        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t
            
            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = self.refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2

            my_mat_final = np.dot(my_mat, my_mat_2) # refine pose means two matrix multiplication
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

        my_r = quaternion_matrix(my_r)[:3, :3]
        #print(my_t.shape)
        my_t = np.array(my_t)
        #print(my_t.shape)
        #print(my_r.shape)


        target = np.dot(self.scaled, my_r.T)
        target = np.add(target, my_t)

        p0 = (int((target[0][0]/ target[0][2])*self.cam_fx + self.cam_cx),  int((target[0][1]/ target[0][2])*self.cam_fy + self.cam_cy))
        p1 = (int((target[1][0]/ target[1][2])*self.cam_fx + self.cam_cx),  int((target[1][1]/ target[1][2])*self.cam_fy + self.cam_cy))
        p2 = (int((target[2][0]/ target[2][2])*self.cam_fx + self.cam_cx),  int((target[2][1]/ target[2][2])*self.cam_fy + self.cam_cy))
        p3 = (int((target[3][0]/ target[3][2])*self.cam_fx + self.cam_cx),  int((target[3][1]/ target[3][2])*self.cam_fy + self.cam_cy))
        p4 = (int((target[4][0]/ target[4][2])*self.cam_fx + self.cam_cx),  int((target[4][1]/ target[4][2])*self.cam_fy + self.cam_cy))
        p5 = (int((target[5][0]/ target[5][2])*self.cam_fx + self.cam_cx),  int((target[5][1]/ target[5][2])*self.cam_fy + self.cam_cy))
        p6 = (int((target[6][0]/ target[6][2])*self.cam_fx + self.cam_cx),  int((target[6][1]/ target[6][2])*self.cam_fy + self.cam_cy))
        p7 = (int((target[7][0]/ target[7][2])*self.cam_fx + self.cam_cx),  int((target[7][1]/ target[7][2])*self.cam_fy + self.cam_cy))
        
        cv2.line(rgb_original, p0,p1,(255,255,255), 2)
        cv2.line(rgb_original, p0,p3,(255,255,255), 2)
        cv2.line(rgb_original, p0,p4,(255,255,255), 2)
        cv2.line(rgb_original, p1,p2,(255,255,255), 2)
        cv2.line(rgb_original, p1,p5,(255,255,255), 2)
        cv2.line(rgb_original, p2,p3,(255,255,255), 2)
        cv2.line(rgb_original, p2,p6,(255,255,255), 2)
        cv2.line(rgb_original, p3,p7,(255,255,255), 2)
        cv2.line(rgb_original, p4,p5,(255,255,255), 2)
        cv2.line(rgb_original, p4,p7,(255,255,255), 2)
        cv2.line(rgb_original, p5,p6,(255,255,255), 2)
        cv2.line(rgb_original, p6,p7,(255,255,255), 2)
        
        #print('estimated rotation is :{0}'.format(my_r))
        #print('estimated translation is :{0}'.format(my_t))

        #time2 = time.time()
        #print('inference time is :{0}'.format(time2-time1))
        cv2.imshow('rgb', cv2.cvtColor(rgb_original, cv2.COLOR_BGR2RGB))  # OpenCV uses BGR model
        cv2.waitKey(1) # pass any integr except 0, as 0 will freeze the display windows 


def main(args):
    seg_model = segnet()
    seg_model.cuda()

    #seg_model.load_state_dict(torch.load('./segschaltgabel.pth'))     #uncomment if we are using the original segnet 
    #seg_model.load_state_dict(torch.load('./schaltgabel_pruned.pth')) 
    #seg_model.load_state_dict(torch.load('./stift_pruned.pth'))
    #seg_model.load_state_dict(torch.load('./flansch_pruned.pth'))

    if opt.model =='flansch':
        seg_model.load_state_dict(torch.load('./segflansch.pth'))
        #seg_model.load_state_dict(torch.load('./flansch_pruned.pth'))
        idx =0
        scaled  = np.array([[59, -42, 59], [59,-42, -59], [59, 20, -59], [59,20,59],
                           [-59, -42, 59], [-59,-42, -59],[-59, 20, -59],[-59,20,59]])/1000                               #flansch_3d_bbox
    elif opt.model == 'schaltgabel':
        seg_model.load_state_dict(torch.load('./segschaltgabel.pth'))
        #seg_model.load_state_dict(torch.load('./schaltgabel_pruned.pth'))
        idx = 1 
        scaled = np.array([[-54.7478, -16.7500, 23], [-54.7478,-16.7500,0],[54.7478,-16.7500,0],[54.7478,-16.7500,23],
                           [-54.7478, 130.85, 23], [-54.7478,130.85,0],[54.7478,130.85,0],[54.7478,130.85,23]]) /1000     #schaltgabel_3d_bbox 
    else :
        #seg_model.load_state_dict(torch.load('./stift_pruned.pth'))
        seg_model.load_state_dict(torch.load('./segstift.pth'))
        idx = 2
        scaled = np.array([[-20,-35,19.9878],[-20,-35,-19.9878],[20,-35,-19.9878],[20,-35,19.9878],
                          [-20,87,19.9878],[-20,87,-19.9878],[20,87,-19.9878],[20,87,19.9878]])/ 1000                     #stift_3d_bbox 
       

    seg_model.eval()
    estimator = PoseNet(num_points, num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points, num_objects)
    refiner.cuda()
    estimator.load_state_dict(torch.load('./pose_model_current.pth'))
    refiner.load_state_dict(torch.load('./pose_refine_model_current.pth'))
    estimator.eval()
    refiner.eval()


    pe = pose_estimation(seg_model, estimator, refiner, idx, scaled) 
    rospy.init_node('pose_estimation', anonymous= True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ('Shutting down ROS pose estimation module')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

'''
how to resize the display window via trackbar:
https://answers.ros.org/question/257440/python-opencv-namedwindow-and-imshow-freeze/
'''
