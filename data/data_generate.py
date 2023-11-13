import json, random, os
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from tqdm import tqdm

def generate_color():
    '''生成颜色'''
    min_brightness = 150
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    brightness = (random.randint(min_brightness, 255))/(r+g+b)
    b *= brightness
    g *= brightness
    r *= brightness
    return (int(b), int(g), int(r))

def color(img):
    '''渐变色'''
    def Make_gradation_img_data(width, height, rgb_start, rgb_stop, horizontal=(True, True, True)):
        '''Make gradation image data'''
        result = np.zeros((height, width, 3), dtype=np.uint8)
        for i, (m,n,o) in enumerate(zip(rgb_start, rgb_stop, horizontal)):
            if o:
                result[:,:,i] = np.tile(np.linspace(m, n, width), (height, 1))
            else:
                result[:,:,i] = np.tile(np.linspace(m, n, width), (height, 1)).T
        return result

    MakeGradationImg = lambda width, height, rgb_start, rgb_stop, horizontal=(True, True, True):Image.fromarray(Make_gradation_img_data(width, height, rgb_start, rgb_stop, horizontal))

    # 生成渐变色图片
    raw_size = 70
    gra_img = MakeGradationImg(raw_size, raw_size, generate_color(), generate_color(), (True, True, True))
    gra_img = gra_img.rotate(45)
    # box = ((raw_size-40)/2, (raw_size-40)/2, (raw_size-40)/2+40, (raw_size-40)/2+40)
    box = ((raw_size-40)/2, (raw_size-40)/2, (raw_size-40)/2+img.size[0], (raw_size-40)/2+img.size[1])
    gra_img = gra_img.crop(box)
    gra_img = gra_img.convert('RGBA')

    # 抠出汉字
    gra_pixel = gra_img.load()
    img_pixel = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if img_pixel[x, y][3]!=0:
                gra_pixel[x, y] = (gra_pixel[x, y][0], gra_pixel[x, y][1], gra_pixel[x, y][2], 255)
            else:
                gra_pixel[x, y] = (gra_pixel[x, y][0], gra_pixel[x, y][1], gra_pixel[x, y][2], 0)

    return gra_img
    
def mess(img):
    '''给图像加上噪点'''
    img_pixels = img.load()
    noise_energy = 10
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if img_pixels[x, y][3]!=0:
                img_pixels[x, y] = (img_pixels[x, y][0]+random.randint(-noise_energy, noise_energy), img_pixels[x, y][1]+random.randint(-noise_energy, noise_energy), img_pixels[x, y][2]+random.randint(-noise_energy, noise_energy), 255)
    return img

def shadow(hanzi_img):
    '''向左上方投影1像素'''
    shadow = Image.new('RGBA', hanzi_img.size, (0, 0, 0, 0))
    shift_pos = 1
    hanziz_pixels = hanzi_img.load()
    shadow_pixels = shadow.load()
    for x in range(hanzi_img.size[0]):
        for y in range(hanzi_img.size[1]):
            if hanziz_pixels[x, y][3]!=0:
                if x>=shift_pos and y>=shift_pos:
                    shadow_pixels[x-shift_pos, y-shift_pos] = (int(hanziz_pixels[x, y][0]/4), int(hanziz_pixels[x, y][1]/4), int(hanziz_pixels[x, y][2]/4), 255)
    
    shadow.paste(hanzi_img, (0, 0), mask=hanzi_img)
    return shadow

def noise_background(img):
    # 小点
    draw = ImageDraw.Draw(img, mode='RGBA')
    # 第一个参数：表示坐标
    # 第二个参数：表示颜色
    for i in range(random.randint(0, 50)):
        draw.point([random.randint(0, img.size[0]-1), random.randint(0, img.size[1]-1)], fill=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))

    # 线条
    # 第一个参数：表示起始坐标和结束坐标
    # 第二个参数：表示颜色
    draw.line((random.randint(0, img.size[0]-1), random.randint(0, img.size[1]-1), random.randint(0, img.size[0]-1), random.randint(0, img.size[1]-1)), fill=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))

    # 圆圈
    # 第一个参数：表示起始坐标和结束坐标（圆要画在其中间）
    # 第二个参数：表示开始角度
    # 第三个参数：表示结束角度
    # 第四个参数：表示颜色
    y0 = random.randint(0, img.size[0]-1)
    x0 = random.randint(0, img.size[1]-1)
    draw.arc((y0, x0, random.randint(y0, img.size[0]-1), random.randint(x0, img.size[1]-1)),0,360,fill=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))

    return img




def generate_pic(output_index):
    # 参数
    default_size = 40 # 字体默认大小
    character_num = 4 # 四个字符
    pic_size = [3, default_size*(character_num+1), default_size+10] # 图片大小
    ttf_path = r"/home/liangzida/workspace/Network_Practice/TTFs/"
    save_path = r"/home/liangzida/workspace/Network_Practice/dataset/"
    default_encoding = "unic" # 默认编码方式
    rotate_max_angle = 5 # 最大旋转角度

    # 读取不同字体
    fonts = []
    ttf_names = os.listdir(ttf_path)
    for ttf_name in ttf_names:
        if ttf_name==".DS_Store":
            continue
        fonts.append(ImageFont.truetype(ttf_path+ttf_name, default_size, encoding=default_encoding))
    character_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']

    # 背景图
    full_img = Image.new('RGBA', (pic_size[1], pic_size[2]), (255, 255, 255, 255))
    full_img = noise_background(full_img)
    
    # 生成字符
    characters = []
    character_fonts = [] # 字体
    character_rotates = [] # 旋转度数
    character_poses = [] # 位置
    for i in range(character_num):
        img = Image.new('RGBA', (default_size, default_size), (0, 0, 0, 0))
        # 生成随机字符
        character = character_list[random.randint(0, len(character_list)-1)]
        characters.append(character)
        # 随机选择字体
        character_fonts.append(random.randint(0, len(fonts)-1))
        # 绘制
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), characters[i], fill=generate_color(), font=fonts[character_fonts[i]])
        # 渐变色
        img = color(img)
        # 旋转
        character_rotates.append(random.randint(-rotate_max_angle, rotate_max_angle))
        img = img.rotate(character_rotates[i])
        # 阴影
        img = shadow(img)
        # 加噪点
        img = mess(img)
        # 覆盖图片
        tmp = (default_size*(i+1), 0)
        character_poses.append(tmp)
        full_img.paste(img, character_poses[i], mask=img)
    save_path += str(output_index)
    for character in characters:
        save_path += ("_" + character)
    # for character_font in character_fonts:
    #     save_path += ("_" + str(character_font))
    save_path += ".jpg"
    full_img = full_img.convert("RGB")
    full_img.save(save_path)


if 1:
    datasetsize = 200000
    for index in tqdm(range(datasetsize)):
        generate_pic(index)