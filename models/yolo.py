import torch. nn as nn
import torch

achitecture_config = [(7, 64, 2, 3),
    "M",                  # maxpooling
    (3, 192, 1, 1),
    "M",                  # maxpooling
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",                  # maxpooling
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],       # tuples and number of repeats=4
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",                  # maxpooling
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],       # tuples and number of repeats=2
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = achitecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [ CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]) ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [ nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) ]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [ CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]) ]
                    layers += [ CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]) ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )

def rename_keys(checkpoint):
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace('_orig_mod.darknet.', '')  # Loại bỏ tiền tố "_orig_mod."
        new_state_dict[new_key] = v
    return new_state_dict
def test(S =7, B = 2 , C = 3):
    model = Yolov1(split_size= S, num_boxes= B, num_classes= C)
    model = torch.compile(model)
    pretrain_weight = '/home/chaos/Documents/ChaosAIVision/temp_folder/backbone448/weights/last.pt'
    checkpoint = torch.load(pretrain_weight)
    backbone_state_dict = rename_keys(checkpoint['model_state_dict'])
    # for key in backbone_state_dict.keys():
    #     print(key)
    load_result = model.darknet.load_state_dict(backbone_state_dict, strict=False)
    for key in load_result.unexpected_keys:
        print(f" - {key}")







   

    # model = torch.compile(model)
    # model.darknet.load_state_dict(backbone_state_dict, strict=False)
    # for param in model.darknet.parameters():
    #     param.requires_grad = False

    # # Kiểm tra xem các tham số của backbone đã được đóng băng chưa
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")


    x = torch.randn((2, 3, 448, 448))


    x = torch.randn((2, 3, 448,448))
    print(model(x).shape)
if __name__ == "__main__":
    test()

                    

                       