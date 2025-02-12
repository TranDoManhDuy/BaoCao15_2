import torch
import torch.nn as nn

# Kiến trúc mạng.
architecture_config = [
    # (kernel_size, out_channels, stride, padding)
    (7, 64, 2, 3),
    "M", # M maxplool(kernel_size = 2, stride = 2)
    (3, 192, 1, 1),
    "M", 
    (1, 128, 1, 0),
    (3, 256, 1, 0),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List
[(1, 256, 1, 0), (3, 512, 1, 1), 4], # (), (), n trong đó mỗi
    # ngoặc là 1 lớp và n là số lần lặp lại.
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]
# không bao gồm các lớp FC.

# xây dựng lên một nơ ron mạng.
class CNNBlock(nn.Module):
    # in_channels và out_channels xác định đầu vào, đầu ra của lớp Conv2D trọng mạng nơ ron tích chập.
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) # kwargs chứa các tham số bổ sung.
        self.batchnorm = nn.BatchNorm2d(out_channels) # vốn dĩ ko nằm trong yoloV1, được áp dụng từ yolov2. Chính vì sử dụng batchnorm nên ko cần trọng số bias
        # hàm kích hoạt.
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    # trình từ đi qua nơ ron.
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class Yolov1(nn.Module):
    # mọi thứ đều phải kế thừa nn.Module, nơi chức nhiều các hàm, phương thức phục vụ cho việc xây dựng mô hình.
    def __init__(self, in_channels = 3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        # darknet là thành phần chính của mô hình, trích xuất đặc trưng từ đầu vào.
        self.darknet = self._create_conv_layers(self.architecture)
        # lớp fully connection
        self.fcs = self._create_fcs(**kwargs)
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    # xây dựng backbone.
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(in_channels, out_channels=x[1], kernel_size = x[0], stride=x[2], padding=x[3])
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(in_channels, 
                                 conv1[1], 
                                 kernel_size = conv1[0], 
                                 stride = conv1[2], 
                                 padding=conv1[3])
                    ]
                    
                    layers += [
                        CNNBlock(conv1[1], 
                                 conv2[1], 
                                 kernel_size = conv2[0], 
                                 stride=conv2[2], 
                                 padding = conv2[3])
                    ]
                    
                    in_channels = conv2[1]
        # là một container của pytorh, gói các layer lại với nhau và thực hiện một cách tuần tự.
        return nn.Sequential(*layers)

    # Tạo ra lớp fully connected để xử lý các đặc trưng đã được trích xuất, từ đố đưa ra dự đoán.
    def _create_fcs(self, split_size, num_boxes, num_classes):
        return nn.Sequential(
            nn.Flatten(), # làm phẳng tensor
            nn.Linear(1024 * split_size * split_size, 4096), # kết nối đầu ra của flatten với 4096 node.
            # giảm số lượng node giúp mô hình tập trung vào các đặc trưng hơn, giảm lượng dữ liệu tránh overfiting.
            # tăng khả năng tổng quát.
            nn.Dropout(0.0), # là một kí thuật regulazization nhằm giảm overfiting
            nn.LeakyReLU(0.1), # hàm kích hoạt, biến thể của ReLu loại bỏ nguy cơ chết nơ ron
            nn.Linear(4096, split_size * split_size * (num_classes + num_boxes * 5)), # (S, S, 30) where C + B + 5 = 30
        )
