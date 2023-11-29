# Reidentification-and-Tracking

![demo](https://github.com/RayCali/Reidentification-and-Tracking/assets/90102246/08527403-4b0a-47ca-a613-f9a9c68e4db3)

Keywords: Pytorch, Reidentification, Tracking, ROS, Triplet mining, Yolo

Contains code for training your own reidentification model with CUHK03 and a ROS package to perform realtime reidentification and tracking with a camera.

The training uses Hard Triplet Mining like in the paper "In Defense of the Triplet Loss for Person Re-Identification". Link to the paper: https://paperswithcode.com/paper/in-defense-of-the-triplet-loss-for-person-re

## Link to download the training and validation set as well as the best achieved model:
https://drive.google.com/drive/u/0/folders/1ZxXXPJLHg_2lZutzrUJEhpk7deBiY1x2

The model managed to achieve a Rank 1 accuracy of 89% on the validation set.

Remember to have the train and val folders and the model in the "reid" folder of this git and inside the src that is inside the "src/reidentification" folder if you want to try tracking in ROS.


## Tracking
Tracking in ROS works by having YOLOv8 as a person detector and the reidentification network as the tracker/reidentifier. The first person YOLO detects will be the target. The topic that the camera image should be published to is "/camera/image".
To run the node run "rosrun reidentification reid_nodev3.py". The node will also publish the same image with the target being marked to "/reid/image" 
