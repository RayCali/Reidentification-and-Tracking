import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from TripletLoss import *
from Network import *
from Data_handler import *

traininglosslist = []


def main():
    global traininglosslist, validationlosslist, height, width
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    siamese_network = SiameseNetwork().to(device)
    siamese_network.train()
    criterion = TripletLoss(margin=1).cuda()
    lr = 0.00005
    optimizer = torch.optim.Adam(siamese_network.parameters(), lr=lr, betas=(0.9, 0.999))
    num_epochs = 10
    nr_of_classes = 736
    batch_size = 72
    done_once = False
    peek_size = 120
    train_or_eval = input("Do you want to train, evaluate or test? (train/eval/test)")
    save_path = input("Enter name for saving model or loading model: ")

    if train_or_eval == 'train':
        images, labellist = get_labels(device, nr_of_classes)
        valimages, vallabels = get_valdata(device)
        losslist = [1]
        for epoch in range(num_epochs):
            if epoch % 1 == 0:
                print("Epoch: ", epoch)
                mean_loss = np.mean(losslist)
                print("Avg loss: ", mean_loss)
                losslist = []
                eval(valimages, vallabels, siamese_network, device, False, peek_size)
                siamese_network.train()
                if epoch > 50:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * (0.001 ** ((epoch - 50) / 50.0))
                    if not done_once:
                        for param_group in optimizer.param_groups:
                            param_group['betas'] = (0.5, 0.999)
                        done_once = True
            for i in range(int(len(labellist)/batch_size)):
                currentimages = images[i*batch_size:(i+1)*batch_size].to(device)
                currentlabels = torch.Tensor(labellist[i*batch_size:(i+1)*batch_size]).to(device)
                output = siamese_network(currentimages)
                loss = criterion(output, currentlabels)
                loss[0].backward()
                losslist.append(loss[0].item())
                optimizer.step()
                optimizer.zero_grad()
            images, labellist = shuffle_dem_tings(images, nr_of_classes, device)
            traininglosslist.append(np.mean(losslist))
        torch.save(siamese_network.state_dict(), save_path)
        plt.plot(traininglosslist)
        plt.plot(validationlosslist)
        print("Model saved")
        plt.legend(['Training loss', 'Validation error'], loc='upper right')
        plt.show()


    
    elif train_or_eval == "eval":
        print("loading valimages to find optimal threshold")
        valimages, vallabels = get_valdata(device)
        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load(save_path))
        ok = eval(valimages, vallabels, model, device, True, 120)
        th = find_optimal_threshold(model, device, valimages, vallabels)

    elif train_or_eval == "test":
        print("loading valimages to find optimal threshold")
        valimages, vallabels = get_valdata(device)
        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load(save_path))
        th = find_optimal_threshold(model, device, valimages, vallabels)
        convert_tensor = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229,0.224,0.225])])
        while True:

            print("Enter name to image 1 in the val folder (exit to stop): ")
            path1 = input()
            if path1 == "exit":
                break
            print("Enter path to image 2 in the val folder (exit to stop): ")
            path2 = input()
            if path2 == "exit":
                break

            img1 = convert_tensor(Image.open("val/"+path1+".jpg")).unsqueeze(0).to(device)
            img2 = convert_tensor(Image.open("val/" + path2 + ".jpg")).unsqueeze(0).to(device)
            out1 = model(img1)
            out2 = model(img2)
            dist = F.pairwise_distance(out1, out2)
            if dist < th:
                print("Same person")
            else:
                print("Not same person")




        

main()