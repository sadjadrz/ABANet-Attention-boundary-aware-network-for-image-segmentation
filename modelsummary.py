from model import ABANet
import  pytorch_model_summary
from torchsummary import summary

net = ABANet(3, 1)
net.cuda()
summary(net,(3,224,224))
