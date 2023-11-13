import os.path
from PIL import Image
import torchvision.transforms as standard_transforms
import torch
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from torch.utils.data import DataLoader # DataLoader需实例化，用于加载数据
from tqdm import tqdm

class MyDataset(Dataset):   # 继承Dataset类
    def __init__(self, device, dataset_dir_path, dataset_size):
        def name_to_int(character):
            character = character[0]
            if character>='a' and character<='z':
                return ord(character)-ord('a')
            if character>='A' and character<='Z':
                return ord(character)-ord('A')+26
            if character>='0' and character<='9':
                return ord(character)-ord('0')+52
        # 数据集图片的路径
        paths = os.listdir(dataset_dir_path)
        # PIL Image 转 Tensor
        transform = standard_transforms.ToTensor()
        dataset = []
        labelset = []
        for index in tqdm(range(min(dataset_size,len(paths)))):
            # 从 paths[] 中的文件名中提取出坐标
            tmp = paths[index][0:-4].split('_')
            label = torch.zeros(4).long()
            label[0] = name_to_int(tmp[1])
            label[1] = name_to_int(tmp[2])
            label[2] = name_to_int(tmp[3])
            label[3] = name_to_int(tmp[4])
            labelset.append(label)
            # 将图片转化为 tensor
            image = Image.open(dataset_dir_path+"/"+paths[index])
            image_tensor = transform(image)
            image_tensor = torch.nn.functional.normalize(image_tensor, dim=2)
            dataset.append(image_tensor)
        # 合并两个 list 为 Tensor
        dataset = torch.stack(dataset)#.to(device)
        labelset = torch.stack(labelset)#.to(device)
        
        # 数据和标签
        self.x_data = dataset
        self.y_data = labelset
        # 数据集的长度
        self.length = len(self.y_data)
        
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self): 
        return self.length

def get_dataloader(batch_size, dataset_dir_path, shuffle=True, num_workers=0, device='cuda', dataset_size=20000):
    # 实例化
    my_dataset = MyDataset(dataset_dir_path=dataset_dir_path, device=device, dataset_size=dataset_size) 
    data_loader = DataLoader(dataset=my_dataset, # 要传递的数据集
                            batch_size=batch_size, #一个小批量数据的大小是多少
                            shuffle=shuffle, # 数据集顺序是否要打乱，一般是要的。测试数据集一般没必要
                            num_workers=num_workers) # 需要几个进程来一次性读取这个小批量数据，默认0，一般用0就够了，多了有时会出一些底层错误。
    return data_loader
