# ma_densefusion
This repository contains Mr. Yue Huang's work in scope of his master thesis, during which he implemented an object segmentation and 6d-pose estimation system for industrial metallic parts targeting at robotics Bin-Picking tasks based on the [DenseFusion](https://arxiv.org/abs/1901.04780) paper.

## ma_densefusion Overview
The ma_densefusion system is capable of segmenting a texturelose part from it's background(usually a blue box) and generating a mask, the mask is then feed into an iterative pose estimation network adapted from the original [implementation](https://github.com/j96w/DenseFusion). The two separate pipelines(i.e., segmentation and pose estimation) can be combined together and run in real time with an inference time of around 0.08s, both for single and mutiple objects scenario. In case of multiple objects scenario, the mask with the maximal area is simply chosen as the final mask to be feed into the pose estimation network. As the roboter arm grips one part at once for Bin-Picking applications, the "maximal area" strategy should be enough.

## Directory Structure
* **datasets**
    * **data**
        * **01** rgb,mask,depth,ply,pcd,train/test indices and ground truth poses for follower(object id:0)
        * **02** rgb,mask,depth,ply,pcd,train/test indices and ground truth poses for shifting fork(object id:1)
        * **03** rgb,mask,depth,ply,pcd,train/test indices and ground truth poses for shifting rod(object id:2)
    * **models**: CAD models of the objects in our experiments
    * **dataset.py**: Dataset loader for all three objects
    * **plygenerator.py**: Python script for generating point cloud in .ply format using a pair of aligned depth and color png files, taken from [here](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/generate_pointcloud.py)
    * **plybatchgenerator.py**: Python script for generating point cloud in .ply format for all pairs of depth and rgb png files in a folder
    * **ply2pcd.sh**:Bash script for converting all .ply files to .pcd files, this requires installation of the [PCL](http://pointclouds.org/) library
* **densefusion_ros**: single-node ros package that subscribes depth and rgb topics for segmentation and pose estimation utilizing the pre-trained models
  * **src**
    * **densefusion_ros.py**:ros node for segmentation and pose estimation in real time
    * **segnet.py**:pruned version of semantic segmentation network(Encoder-Decoder)
    * **segnet_original.py**:original version of semantic segmentation network(Encoder-Decoder)


* **experiments**:useful scripts for model training and evalution
* **generategtfromposedata**: scripts for generating ground truth poses from datatset for evalution using vsd, adi, recall rate metrics
    * **pred_1.txt**:model predicted poses for follower using mask ouputed by original version of semantic segmenation network
    * **pred_2.txt**:model predicted poses for shifting fork using mask ouputed by original version of semantic segmenation network
    * **pred_3.txt**:model predicted poses for shifting rod using mask ouputed by original version of semantic segmenation network
    * **generategt.py**:Python script for generating ground truth poses in the format that our evaluation pipeline accepts from original datasets
    * **posedata_1.yml**:original ground truth pose data for follower, same as /data/01/posedata.yml
    * **posedata_3.yml**:original ground truth pose data for shifting rod, same as /data/03/posedata.yml

* **lib**:
    * **lib/loss.py**: Loss calculation for DenseFusion model.
	* **lib/loss_refiner.py**: Loss calculation for iterative refinement model.
	* **lib/transformations.py**: [Transformation Function Library](https://www.lfd.uci.edu/~gohlke/code/transformations.py.html).
    * **lib/network.py**: Network architecture.
    * **lib/extractors.py**: Encoder network architecture adapted from [pspnet-pytorch](https://github.com/Lextal/pspnet-pytorch).
    * **lib/pspnet.py**: Decoder network architecture.
    * **lib/utils.py**: Logger code.
    * **lib/knn/**: CUDA K-nearest neighbours library adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda).
+ **man_stl**: CAD models in .stl format
* **seg**:original and pruned version of semantic segmentation network
    * **segmentation**: datastes for segmentation and train/test indices
        * **writeline.py**: scripts for creating train/test index files
        * **mask**: mask daatstes for semantic segmentation
        * **rgb**:  rgb datasets for semantic segmentation
        * **seg_result** :semantic segmentation results generated with "/seg/segeval.py"
        * **txt_files**:train and test indicies files
    * **segeval.py**: evalution scripts for generating segmentation results
    * **segnet.py**:original semantic segmentation network from DenseFusion
    * **loss.py**:Cross Entropy loss used for semantic segmentation network
    * **train.py**: Training scripts for semantic segmentation network
    * **data_controller.py**: Dataloder for semantic segmentation datasets

* **tools**
	* **tools/_init_paths.py**: Add local path.
    * **tools/eval_adi.py**: scripts for evaluating on adi,vsd metrics, still needs to be modified
	* **tools/eval_poseownnet.py**: Evaluation code for own posent dataset.
	* **tools/train.py**: Training code for own posenet dataset.
* **trained_models**: pre-trained models,which includes a pose estimation network and a refiner network model
* **useful**:some useful scripts that save you a lot of dirty work like starting docker and camera, republishing ros topics to [LCM](https://lcm-proj.github.io/index.html) format, converting datasets from LabelFusion format to our format, etc.
* **Paper.pdf**: An example paper from previous work

## Datasets Downloading
The Datasets used in this project can be downloaded from [here](https://drive.google.com/open?id=1k8muuXmz4wddMxDQou6hEJGaLqHojiNU). After downloading, move the folders named 01,02,03 to the datasets/data folder and the .ply files to the datasets/models folder, then you can start to train or evaluate. Note that you may need to delete the original subfolders in datasets/data folders(i.e., 01,02,03), they are actually the same as their counterparts in the downloaded datasets.

## Creating Your Own Datasets
The [LabelFusion](https://github.com/RobotLocomotion/LabelFusion) is an awesome tool for creating own datastes. It was designed, however,to accept as input messages in the LCM message format that matches the [openni2-camera.lcm](https://github.com/openhumanoids/openni2-camera-lcm) driver. An intermidate layer,such as the [rgbd_ros_to_lcm](https://github.com/MobileManipulation/rgbd_ros_to_lcm) ros package,is necessary when the pipeline is used with a RealSense cmaera.

There are also some necessary modifications in the original LabelFusion implementation to accept different camera intrinsic parameters:  

1.For rendering training image, edit the "setCameraIntrinsicsAsus" function at the bottom of the file ```LabelFusion/modules/labelfusion/rendertrainingimages.py``` to the the following:
```python
def setCameraInstrinsicsAsus(view):
        principalX = 315.2859903333336
        principalY = 244.88168334960938
        focalLength = 616.0936279296875
        setCameraIntrinsics(view, principalX, principalY, focalLength)
```
2.For ElasticFusion to reconstruct 3D scene correctly, change line 37 of the file ```LabelFusion/scripts/prepareForObjectAlignment.py``` to the following:
```python
os.system(path_to_ElasticFusion_executable + " -l ./" + lcmlog_filename+ " -cal ./camera.cfg")
```
3.Create a camera.cfg file in your lcm-log folder with the camera instrinsic infomation ```fx fy cx cy``` in just one line.

The rest is the same as what is documented in LabelFusion's original pipeline. The point cloud of .ply format can be obtained by using [this](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/generate_pointcloud.py) python script. If you need a .pcd format point cloud absolutely,try the "pcl_converter" tool after installing the [PCL](http://pointclouds.org/) library, the syntax is as follows:
```bash
pcl_converter -f ascii 0000.ply 0000.pcd
```
You may encounter a "Cannot read geometry" error, this is due to the fact that the .ply file does not declare any information about faces(just vertices).A workaround is to add the following two lines after the line "property uchar alpha" in the header part of the .ply file.
```
element face 0
property list uchar int vertex_indices
```

## Training
In the /ma_densefusion folder, run:
```
./experiments/scripts/train_poseownnet.sh
```
## Evaluation
In the /ma_densefusion folder, run:
```
./experiments/scripts/eval_poseownnet.sh
```
## Training of Semantic Segmentation Network
In the /ma_densefusion/seg/ folder, run:
```
python3 train.py
```
## Getting Segmentation Results
In the /ma_densefusion/seg/ folder, run:
```
python3 segeval.py
```
Note that you may need to adjust some lines of code to get segmentation results for just one picture or the whole datasets according to your need, see comments in the python script for detail.
## Starting ROS Node
Copy the /densefusion_ros folder to your ROS catkin_ws/src, soure your catkin_ws/devel/setup.bash file,change directory to catkin_ws/src/densefusion_ros/src, and then run:
```
rosrun densefusion_ros densefusion_ros.py --model=flansch
```
the option "--model" allows you to specify which object you are woking with, change "--model=flansch" to "--model=schaltgabel" or "--model=stift" if you want to detect other objects.
## Some Demo Pictures
![Result for follower](./3dbbox_flansch)
![Result for shifting rod](./3dbbox_stift)




