import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

eps = 1e-8
accmin = 0
earl_stop_iter = 0
traininglosslist = []
validationlosslist = []
totallistofimages = []
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # weights = ResNet18_Weights.DEFAULT
        # self.preprocess = weights.transform()
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last classification layer
        self.feat = nn.Linear(self.fc_in_features, 1024)
        self.feat_bn = nn.BatchNorm1d(1024)
        self.fc = nn.Linear(1024, 128)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.feat_bn.weight, 1)
        nn.init.constant_(self.feat_bn.bias, 0)
        nn.init.kaiming_normal_(self.feat.weight, mode='fan_out')
        nn.init.constant_(self.feat.bias, 0)


    def forward_one(self, x):
        x = self.resnet(x)
        x = x.view(x.size()[0], -1)
        x = self.feat(x)
        x = self.feat_bn(x)
        x = F.relu(x)
        x = self.fc(x)

  

        return x

    def forward(self, input1):

        output1 = self.forward_one(input1)
        return output1
    def inference(self, input1, input2):
        output1 = self.forward_one(input1)

        output2 = self.forward_one(input2)
        
        distance = F.pairwise_distance(output1, output2, p=2)
        return distance


