#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import numpy as np

from Network import SiameseNetwork
import torch
import cv2
from cv_bridge import CvBridge
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image as PILImage
import torch.nn.functional as F
import time

def yolo_callback(msg: Image):
    global sub_img, reid_model, yolo_model,height,width, target_embeddings, count, pub_img, distance, save_video, out, start_time, max_count, image_count, got_target, target_embedding, threshold
    convert_tensor = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229,0.224,0.225])
        ])
    bridge = CvBridge()
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    results = yolo_model.predict(cv_img, classes=0, verbose=False)
    results = results[0]
    if not got_target:
        target = results.boxes[0]
        coordinates = target.xyxy[0]
        targetboundingbox = cv_img[int(coordinates[1]):int(coordinates[3]), int(coordinates[0]):int(coordinates[2])]
        #cv to pil
        targetboundingbox = PILImage.fromarray(targetboundingbox)
        #show
        targetboundingbox.show()
        targetboundingbox = convert_tensor(targetboundingbox)
        target_embedding = reid_model(targetboundingbox.unsqueeze(0))
        target_embeddings.append(target_embedding)
        got_target = True
    found = False
    for b in results.boxes:
        box = b.xyxy[0]
        boundingbox = cv_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        boundingbox = PILImage.fromarray(boundingbox)
        boundingbox = convert_tensor(boundingbox)
        embedding = reid_model(boundingbox.unsqueeze(0))
        distance = F.pairwise_distance(embedding, target_embedding)
        distance_cont = []
        print(len(target_embeddings))
        for target_embed in target_embeddings:
            distance_cont.append(F.pairwise_distance(embedding, target_embed))
        distance_cont = sum(distance_cont)/len(distance_cont)
        rospy.loginfo("Distance: " + str(distance))
        if distance < threshold or distance_cont < threshold:
            rospy.loginfo("Target Found")
            found = True
            if count >= max_count:
                if len(target_embeddings) < image_count:
                    target_embeddings.append(reid_model(boundingbox.unsqueeze(0)))
                else:
                    target_embeddings.pop(0)
                    target_embeddings.append(reid_model(boundingbox.unsqueeze(0)))
                count = 0
            #draw boundingbox on cv_img
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            cv2.rectangle(cv_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(cv_img, "Target Found", org = (100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=2)
            sensor_msg = bridge.cv2_to_imgmsg(cv_img, encoding='passthrough')
            pub_img.publish(sensor_msg)

            break
    if not found:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        sensor_msg = bridge.cv2_to_imgmsg(cv_img, encoding='passthrough')
        pub_img.publish(sensor_msg)

        rospy.loginfo("Target Not Found")
    count += 1

    return

        

        
        





got_target = False
start_time = 0
distance = 0
max_count = 10
count = max_count
image_count = 3
target_embeddings = []
target_embedding = None
sub_img = None
pub_img = rospy.Publisher('/reid/image', Image, queue_size=10)
reid_model = SiameseNetwork()
reid_model.load_state_dict(torch.load('reid_Cust.pt'))
reid_model.eval()
yolo_model = YOLO('yolov8s.pt')
width = 128
height = 256
threshold = 23.8



if __name__ == '__main__':
    rospy.init_node('reid_node', anonymous=True)
    sub_img = rospy.Subscriber('/camera/image', Image, yolo_callback, queue_size=1)
    rospy.spin()