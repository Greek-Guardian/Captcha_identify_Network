import torch
import torch.nn
from PIL import Image
import torchvision.transforms as standard_transforms
import torch

def deploy(opt, model, device):
    def name_to_int(character):
        character = character[0]
        if character>='a' and character<='z':
            return ord(character)-ord('a')
        if character>='A' and character<='Z':
            return ord(character)-ord('A')+26
        if character>='0' and character<='9':
            return ord(character)-ord('0')+52

    # PIL to Tensor
    transform = standard_transforms.ToTensor()
    image = Image.open(opt.pic_dic_path+opt.pic_name)
    image_tensor = transform(image).to(device)
    image_tensor = torch.nn.functional.normalize(image_tensor, dim=2)
    image_tensor.unsqueeze_(0)

    with torch.no_grad():
        # get expected output
        tmp = opt.pic_name[0:-4].split('_')
        print("Expected:", tmp[1], tmp[2], tmp[3], tmp[4])
        target = torch.tensor([name_to_int(tmp[1]), name_to_int(tmp[2]), name_to_int(tmp[3]), name_to_int(tmp[4])]).long().to(device)
        model.eval()
        prediction = model(image_tensor)
        prediction = prediction[0]
        index_prediction = torch.tensor([prediction[:62].argmax().item(),\
                                        prediction[62:124].argmax().item(),\
                                        prediction[124:186].argmax().item(),\
                                        prediction[186:248].argmax().item()\
                                        ]).long().to(device)
        # Loss
        # criterion = torch.nn.CrossEntropyLoss().to(device)
        # print("Loss:", criterion(prediction.view(4, 62), target))
        # print(prediction)
        character_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
        print("Output  :", character_list[index_prediction[0]], end=' ')
        print(character_list[index_prediction[1]], end=' ')
        print(character_list[index_prediction[2]], end=' ')
        print(character_list[index_prediction[3]])
        return target.equal(index_prediction)
