import torch.nn as nn
from research.rlogger import research_logger_info as rlogger

supported_resnets_layer_cnts = [
    18, 50, 101
]
class Resnet(object):
    def __init__(self,src_resnet_name):
        self.name = src_resnet_name.strip()
        self.layer_cnts= int(src_resnet_name.strip().split('_')[-1])
        if self.layer_cnts not in supported_resnets_layer_cnts:
            raise Exception("resnet-%s is not supported !"%(self.layer_cnts))

        self.src_img_data=""

        

    def show_nets(self):
        '''show nets values.
        '''
        print("self.name: ", self.name)
        print("self.layer_cnts: ", self.layer_cnts)

    def forward(self, src_img_data):




def Test_Resnet_50():
    resnet50 = Resnet("resnet_50")
    resnet50.show_nets()
    # print(str(resnet50))

if __name__=='__main__':
    Test_Resnet_50()

