import torch. nn as nn
import torch

achitecture_config = [(7,64,2,3),'M',
                       (3, 192,1,1), 'M',
                       (1,128,1,0),(3,256,1,1),(1,256,1,0),(3,512,1,1),'M',
                       [(1,256,1,0), (3,512,1,1), 4], (1,512,1,0), (3,1024,1,1),'M',
                       [(1,512,1,0), (3,1024,1,1),2],(3,1024,1,1), (3,1024,2,1),
                       (3,1024,1,1), (3,1024,1,1)
                       
                       ]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias= False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)

        return x
    

class YOLOv1(nn.Module):
    def __init__(self, in_channels = 3, **kwargs):
        super(YOLOv1, self).__init__()
        self.achitecture = achitecture_config
        self.in_channels = in_channels
        self.darknet = self.make_block(self.achitecture)
        self.fcs = self.make_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)             
        x = torch.flatten(x, start_dim= 1)
        x = self.fcs(x)

        return x
    

    def make_block(self, architecture):
        layer = []
        in_channels = self.in_channels

        for m in architecture:
            if type(m) == tuple:
                layer += [ CNNBlock(in_channels,m[1], kernel_size = m[0], stride = m[2], padding = m[3],)]
                in_channels = m[1]

            elif type(m) == str:
                layer += [nn.MaxPool2d(kernel_size= 2, stride= 2)]
            
            elif type (m) == list:
                conv1 = m[0]
                conv2 =m[1]
                num_repeats =  m[2]

                for _ in range(num_repeats):
                    layer += [CNNBlock(in_channels, conv1[1], kernel_size = conv1[0], stride = conv1[2], padding = conv1[3],)]
                    layer += [CNNBlock(conv1[1], conv2[1], kernel_size = conv2[0], stride = conv2[2], padding = conv2[3],)]

                    in_channels = conv2[1]
        return nn.Sequential(*layer)
    
    def make_fcs(self, split_size, num_boxes, num_classes):
        S, B , C = split_size, num_boxes, num_classes
        return nn.Sequential(nn.Flatten(),
                              nn.Linear(1024 * S * S, 4096),
                              nn.Dropout(0.5),
                              nn.LeakyReLU(0.1),
                              nn.Linear(4096, S * S * (C + B * 5)),
                              )
def test(S =7, B = 2 , C = 20):
    model = YOLOv1(split_size= S, num_boxes= B, num_classes= C)
    x = torch.randn((2, 3, 448,448))
    print(model(x).shape)
if __name__ == "__main__":
    test()

                    

                       