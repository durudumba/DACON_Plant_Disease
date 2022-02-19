from collections import Counter
import os
import json
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
import shutil

data_path = './data/augmented_train/'
        
# 생성된 객체를 저장해주는 함수
def saveNchange(target_path, image, option):
    idx = target_path.split('/')[-1]
    shutil.copytree(f'{target_path}', f'{target_path}_{option}')
    os.remove(f'{target_path}_{option}/{idx}.jpg')
    image.save(f'{target_path}_{option}/{idx}.jpg')
    
    file_names = os.listdir(f'{target_path}_{option}')
    for name in file_names:
        temp = name.split('.')
        os.rename(f'{target_path}_{option}/{name}', f'{target_path}_{option}/{temp[0]}_{option}.{temp[1]}')

image_ids = os.listdir(data_path)
json_paths = glob(data_path + './*/*.json')

# 이미지 경로 저장
image_dirs = []
image_labels = []

for image_id in image_ids:
    image_dirs.append(f'{data_path}{image_id}/{image_id}.jpg')


# 이미지 라벨 저장
for json_path in json_paths:
    an_meta = json.load(open(json_path))['annotations']
    json_crop = (an_meta['crop'])
    json_disease = (an_meta['disease'])
    json_risk = (an_meta['risk'])
    label = f'{json_crop}_{json_disease}_{json_risk}'
    image_labels.append(label)

# 데이터프레임화
image_datas = list(zip(image_ids, image_labels))


aug_mul = 16
count_labels = Counter(image_labels)
min_label = min(list(count_labels.values()))
aug_labels = [value for value in count_labels.keys() if count_labels[value] < min_label * aug_mul]


# Augmentation
for index in range(len(image_datas)):
    image_data = image_datas[index]
    if image_data[1] in aug_labels:
        ##증강
        image = Image.open(f'./data/augmented_train/{image_data[0]}/{image_data[0]}.jpg')
        target_path = f'./data/augmented_train/{image_data[0]}'

        # 자르기
        if(image.size[0] > image.size[1]):   # 가로로 긴 경우
            image_top = image.crop((0,0, image.size[0]*(3/5), image.size[1]))
            image_mid = image.crop((image.size[0]*(1/5),0, image.size[0]*(4/5), image.size[1]))
            image_bot = image.crop((image.size[0]*(2/5),0, image.size[0], image.size[1]))
        else: # 세로로 긴 경우
            image_top = image.crop((0,0, image.size[0], image.size[1]*(3/5)))
            image_mid = image.crop((0, image.size[0]*(1/5), image.size[0], image.size[1]*(4/5)))
            image_bot = image.crop((0, image.size[0]*(2/5), image.size[0], image.size[1]))
        # 원본 이미지 회전
        image_90 = image.transpose(Image.ROTATE_90) # 90도 회전
        image_180 = image.transpose(Image.ROTATE_180)   # 180도 회전
        image_270 = image.transpose(Image.ROTATE_270)   # 270도 회전
        # crop_top 이미지 회전
        image_top_90 = image_top.transpose(Image.ROTATE_90) # 90도 회전
        image_top_180 = image_top.transpose(Image.ROTATE_180)   # 180도 회전
        image_top_270 = image_top.transpose(Image.ROTATE_270)   # 270도 회전
        # crop_mid 이미지 회전
        image_mid_90 = image_mid.transpose(Image.ROTATE_90) # 90도 회전
        image_mid_180 = image_mid.transpose(Image.ROTATE_180)   # 180도 회전
        image_mid_270 = image_mid.transpose(Image.ROTATE_270)   # 270도 회전
        # crop_bot 이미지 회전
        image_bot_90 = image_bot.transpose(Image.ROTATE_90) # 90도 회전
        image_bot_180 = image_bot.transpose(Image.ROTATE_180)   # 180도 회전
        image_bot_270 = image_bot.transpose(Image.ROTATE_270)   # 270도 회전
        # save images
        saveNchange(target_path, image_top, 'top')
        saveNchange(target_path, image_mid, 'mid')
        saveNchange(target_path, image_bot, 'bot')
        saveNchange(target_path, image_90, '90')
        saveNchange(target_path, image_180,'180')
        saveNchange(target_path, image_270,'270')
        saveNchange(target_path, image_top_90, 'top_90')
        saveNchange(target_path, image_top_180, 'top_180')
        saveNchange(target_path, image_top_270, 'top_270')
        saveNchange(target_path, image_mid_90, 'mid_90')
        saveNchange(target_path, image_mid_180, 'mid_180')
        saveNchange(target_path, image_mid_270, 'mid_270')
        saveNchange(target_path, image_bot_90, 'bot_90')
        saveNchange(target_path, image_bot_180, 'bot_180')
        saveNchange(target_path, image_bot_270, 'bot_270')


    else:
        continue
    
