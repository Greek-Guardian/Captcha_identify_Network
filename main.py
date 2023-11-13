from train import train
from deploy import deploy
import torch, os
import numpy as np
from random import randint

class Config(object):
    # dataloader
    dataset_dir_path = r"./data/dataset"
    dataset_size = 500 # total num: 20000
    num_workers = 4
    shuffle = False
    # train
    train = True
    model_load_path = r"./checkpoints/captcha.pth"
    model_save_path = r"./checkpoints/captcha.pth"
    load_or_not = False # whether load a model or not
    learning_rate = 0.001
    training_epochs = 10
    batch_size = 10
    seed = 777
    gpu = True
    # deploy
    deploy = True
    deploy_num = 50
    pic_dic_path = r"/home/liangzida/workspace/Network_Practice/data/dataset/"
    pic_name = r"/0_U_P_4_V.jpg"


if __name__ == '__main__':
    torch.set_printoptions(threshold=np.inf)
    # import fire
    # fire.Fire()
    opt = Config()
    if opt.train:
        train(opt)
    if opt.deploy:
        paths = os.listdir(opt.pic_dic_path)
        Accuracy = 0
        for index in range(opt.deploy_num):
            opt.pic_name = paths[randint(0, min(opt.dataset_size,len(paths))-1)]
            device = 'cuda' if torch.cuda.is_available() and opt.gpu else 'cpu'
            model = torch.load(opt.model_load_path, map_location=device)
            if deploy(opt, model, device):
                Accuracy += 1
        print("Accuracy:", Accuracy / opt.deploy_num)