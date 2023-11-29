# Reidentification-and-Tracking

![demo](https://github.com/RayCali/Reidentification-and-Tracking/assets/90102246/08527403-4b0a-47ca-a613-f9a9c68e4db3)

Keywords: Pytorch, Reidentification, Tracking, ROS, Triplet mining, Yolo

Contains code for training your own reidentification model with CUHK03 and a src for a ROS package to perform realtime reidentification and tracking with a camera.

The training uses Hard Triplet Mining like in the paper "In Defense of the Triplet Loss for Person Re-Identification". Link to the paper: https://paperswithcode.com/paper/in-defense-of-the-triplet-loss-for-person-re

## Link to download the training and validation set as well as the best achieved model:
https://drive.google.com/drive/u/0/folders/1ZxXXPJLHg_2lZutzrUJEhpk7deBiY1x2

The model managed to achieve a Rank 1 accuracy of 89% on the validation set.

Remember to have the train and val folders and the model in the "reid" folder of this git.
