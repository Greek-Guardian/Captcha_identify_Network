import torch as t
from torch import nn

class captcha(nn.Module):
    def __init__(self) -> None:
        '''
        # default_size = 40 # 字体默认大小
        # character_num = 4 # 四个字符
        # 图片大小
        # pic_size = [3, default_size*(character_num+1), default_size+10] 
        # pic_size = [3, 200, 50]
        '''
        super(captcha, self).__init__()

        self.keep_prob = 0.5
        # N = (W - F + 2P)/S +1
        self.convolution = nn.Sequential(
        # L1 ImgIn shape=(?, 200, 50, 3)
        #    Conv     -> (?, 200, 50, 30)
        #    Pool     -> (?, 100, 25, 30)
            nn.Conv2d(3, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        # L2 ImgIn shape=(?, 100, 25, 30)
        #    Conv      ->(?, 100, 25, 60)
        #    Pool      ->(?, 50, 50, 60)
            nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(60),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        # L3 ImgIn shape=(?, 50, 50, 60)
        #    Conv      ->(?, 50, 50, 120)
        #    Pool      ->(?, 25, 25, 120)
            nn.Conv2d(60, 120, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(120),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.connect = nn.Sequential(
        # L4 FC 25*25*120 inputs -> 75000 outputs
        # self.fc1 = nn.Linear(25 * 25 * 120, 75000, bias=True)
            nn.Linear(21840, 625, bias=True),
            nn.PReLU(),
            nn.Dropout(p = 1 - self.keep_prob),
            # L5 Final FC 625 inputs -> 62 outputs
            nn.Linear(625, 4*62, bias=False),
        )

    def forward(self, input):
        out = self.convolution(input)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        return self.connect(out)
    
def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        # nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias!=None:
            nn.init.constant_(layer.bias, 0.1)

def evaluate(predictions, targets):
    judge = []
    for prediction, target in zip(predictions, targets):
        pred = t.zeros(4).to(target.device)
        pred[0] = prediction[0:62].argmax()
        pred[1] = prediction[62:124].argmax()
        pred[2] = prediction[124:186].argmax()
        pred[3] = prediction[186:248].argmax()
        if (pred).equal(target):
            judge.append(1)
        else:
            judge.append(0)
    return t.Tensor(judge)