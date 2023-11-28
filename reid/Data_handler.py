import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image
from torch.nn import init
from TripletLoss import *
from Network import *
import numpy as np
from torchvision import transforms
totallistofimages = []
totallistofvalimages = []
width = 128
height = 256
validationlosslist = []
def find_optimal_threshold(model, device, valimages, vallabels):
    model.eval()
    outs = np.zeros((len(vallabels), 128))

    for i in range(int(len(vallabels)/60)):
        currentimages = valimages[i*60:(i+1)*60]
        outs[i*60:(i+1)*60] = model(currentimages).cpu().detach().numpy()
        del currentimages
    
    outs = torch.Tensor(outs).to(device)
    dists = da_dist(outs, squared=False)
 
    labels = torch.Tensor(vallabels).to(device)
    labels = labels.unsqueeze(0)
    labels = labels == labels.t()
    relevant_dists = torch.ones(dists.size()).triu(diagonal=1)
    dists = dists[relevant_dists == 1]
    labels = labels[relevant_dists == 1]
    best_threshold = 0.4
    highest_correct = 0
    thresholds= torch.arange(0, 35, 0.1)
    
    for t in thresholds:
        results = dists <= t
        amount_corr = torch.sum(labels == results)
        if amount_corr > highest_correct:
            best_threshold = t
            highest_correct = amount_corr
    print("Optimal Threshold: "+ str(best_threshold.item()))
    print("Full accuracy: " + str(highest_correct.item()/(dists.size(0))))

    return best_threshold.item()


def eval(valimages,vallabels, model, device, verbose, peek_size):
    global earl_stop_iter, accmin, validationlosslist, totallistofvalimages, width, height
    model.eval()
    
    patience = 10
    piclist_size = peek_size // 6
    shufflefrom = totallistofvalimages[0:piclist_size]
    valtensor = torch.empty(piclist_size, 3, height, width).to(device)
    N = 5
    #compute cmc curve
    totalcmc = np.zeros(peek_size)
    localavg = np.zeros(N)
    dist = []
    k=-1
    for i in range(peek_size):
        if i % 6 == 0:
            k += 1
            if verbose:
                print(k)
        img1 = shufflefrom[k][0]
        out1 = model(img1.unsqueeze(0).to(device))
        shufflefrom[k] = shufflefrom[k][1:]
        for l in range(N):
            for j in range(len(shufflefrom)):
                valtensor[j, :, :, :] = random.choice(shufflefrom[j]).unsqueeze(0).to(device)
            out2 = model(valtensor)
            dist = F.pairwise_distance(out2, out1)
            lol = torch.argmin(dist)
            if lol == k:
                localavg[l] = 1
            else:
                localavg[l] = 0
        totalcmc[i] = np.mean(localavg)
        shufflefrom[k] = torch.cat((shufflefrom[k], img1.unsqueeze(0)), 0)
    accuracy = np.mean(totalcmc)
    validationlosslist.append(1-accuracy)
    print("Rank 1 Accuracy: ", accuracy)


def get_labels(device, nr_of_classes):
    global mean, std, totallistofimages, width, height
    multiple = 2
    convert_tensor = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229,0.224,0.225])
    ])

    images = torch.empty(nr_of_classes*12*multiple, 3, height, width)

    labellist = []
    pic_nr = 0
    intensities = [0.3, 0.5, 0.7, 1.2, 1, 1.4, 1.6]
    
    for i in range(0,nr_of_classes):
        amount = 0
        listofimages = []
        if i % 100 == 0:
            print(i)
        for j in range(10): 
            try:    
                    if i < 10:
                        pic = Image.open("train/000"+str(i)+"_0"+str(j)+".jpg")
                    elif i>=10 and i < 100:
                        pic = Image.open("train/00"+str(i)+"_0"+str(j)+".jpg")
                    else:
                        pic = Image.open("train/0"+str(i)+"_0"+str(j)+".jpg")
                    listofimages.append(pic)

                    amount += 1
                    pic_nr += 1
            except FileNotFoundError:
                continue
        for k in range(10-amount):
            listofimages.append(listofimages[k].transpose(Image.FLIP_LEFT_RIGHT))
        listofimages.append(listofimages[4].transpose(Image.FLIP_LEFT_RIGHT))
        listofimages.append(listofimages[5].transpose(Image.FLIP_LEFT_RIGHT))
        for l in range(len(listofimages)):
            for a in range(multiple-1):
                img = listofimages[l].convert("RGB")
                #random element in intensities
                # Apply the tint by increasing color channels
                r,g,b = img.split()
                rgblist = [r,g,b]
                indexes = [0,1,2]
                index1 = random.choice(indexes)
                indexes.remove(index1)
                index2 = random.choice(indexes)

                rgblist[index1] = ImageEnhance.Brightness(rgblist[index1]).enhance(intensities[random.randint(0,len(intensities)-1)])
                rgblist[index2] = ImageEnhance.Brightness(rgblist[index2]).enhance(intensities[random.randint(0,len(intensities)-1)])

                # Merge the modified channels back together
                tinted_image = Image.merge("RGB", (rgblist[0], rgblist[1], rgblist[2]))
                listofimages.append(tinted_image)
        random.shuffle(listofimages)
        totallistofimages.append(listofimages)
        
    nr = 0
    classindex = np.arange(0, nr_of_classes)
    classpicked = np.zeros(nr_of_classes)
    
    for a in range(int(nr_of_classes*12*multiple/4)):

        randomelement = random.choice(classindex)
        
        
        for l in range(int(12*multiple/4)):
            if classpicked[randomelement] == l:
                # if a == 1104:
                #     print(randomelement)
                #     totallistofimages[randomelement][0+l*4].show()
                images[nr, :, :, :] = convert_tensor(totallistofimages[randomelement][0+l*4]).unsqueeze(0)
                images[nr+1, :, :, :] = convert_tensor(totallistofimages[randomelement][1+l*4]).unsqueeze(0)
                images[nr+2, :, :, :] = convert_tensor(totallistofimages[randomelement][2+l*4]).unsqueeze(0)
                images[nr+3, :, :, :] = convert_tensor(totallistofimages[randomelement][3+l*4]).unsqueeze(0)
                labellist.extend([randomelement, randomelement, randomelement, randomelement])
        nr += 4
        classpicked[randomelement] += 1
        if classpicked[randomelement] == 12*multiple/4:
            index_to_delete = np.where(classindex == randomelement)
            classindex = np.delete(classindex, index_to_delete)
    return images, labellist

def shuffle_dem_tings(images, nr_of_classes, device):
    global totallistofimages, width, height

    convert_tensor = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229,0.224,0.225])
    ])
    for i in range(0,nr_of_classes):
        random.shuffle(totallistofimages[i])
    labellist = []
    multiple = 1
    nr = 0
    classindex = np.arange(0, nr_of_classes)
    classpicked = np.zeros(nr_of_classes)
    
    for a in range(int(nr_of_classes*12*multiple/4)):

        randomelement = random.choice(classindex)
        
        
        for l in range(int(12*multiple/4)):
            if classpicked[randomelement] == l:

                images[nr, :, :, :] = convert_tensor(totallistofimages[randomelement][0+l*4]).unsqueeze(0)
                images[nr+1, :, :, :] = convert_tensor(totallistofimages[randomelement][1+l*4]).unsqueeze(0)
                images[nr+2, :, :, :] = convert_tensor(totallistofimages[randomelement][2+l*4]).unsqueeze(0)
                images[nr+3, :, :, :] = convert_tensor(totallistofimages[randomelement][3+l*4]).unsqueeze(0)
                labellist.extend([randomelement, randomelement, randomelement, randomelement])
        nr += 4
        classpicked[randomelement] += 1
        if classpicked[randomelement] == 12*multiple/4:
            index_to_delete = np.where(classindex == randomelement)
            classindex = np.delete(classindex, index_to_delete)
    return images, labellist

def get_valdata(device):
    global mean, std, totallistofvalimages, width, height
    convert_tensor = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229,0.224,0.225])
    ])
    images = torch.empty(100*6, 3, height, width).to(device)
    labellist = []
    pic_nr = 0
    amountlist = []
    totallistofvalimages = []
    for i in range(0,100):
        amount = 0
        for j in range(10):
            if amount == 6:
                break
            try:
                if i < 10:
                    images[pic_nr, :, :, :] = convert_tensor(Image.open("val/000"+str(i)+ "_0" + str(j)+".jpg")).unsqueeze(0)
                    labellist.append(i)
                else:
                    images[pic_nr, :, :, :] = convert_tensor(Image.open("val/00"+str(i)+ "_0" + str(j)+".jpg")).unsqueeze(0)
                    labellist.append(i)
                pic_nr += 1
                amount += 1
            except FileNotFoundError:
                continue
        totallistofvalimages.append(images[pic_nr-6:pic_nr])
        amountlist.append(amount)

   

    return images, labellist