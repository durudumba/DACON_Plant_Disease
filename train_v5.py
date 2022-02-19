## Library
import numpy as np
from glob import glob
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

from CustomDataset_v5 import CustomDataset
from Network_v5 import Network

## Label dictionary
crop = {'1': '딸기', '2': '토마토', '3': '파프리카', '4': '오이', '5': '고추', '6': '시설포도'}
disease = {
    '1': {'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
    '2': {'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
          'b8': '다량원소결핍 (K)'},
    '3': {'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
    '4': {'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
    '5': {'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
    '6': {'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}}
risk = {'1': '초기', '2': '중기', '3': '말기'}
label_description = {}
for key, value in disease.items():
    label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
    for disease_code in value:
        for risk_code in risk:
            label = f'{key}_{disease_code}_{risk_code}'
            label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'
# encoding 할 때 사용할 딕셔너리
label_encoder = {key: idx for idx, key in enumerate(label_description)}
# decoding 할 때 사용할 딕셔너리
label_decoder = {val: key for key, val in label_encoder.items()}


# 데이터 경로 설정
data_path = './data/augmented_train/'
json_paths = glob(data_path + './*/*.json')


# Hyperparameter
batch_size = 32
learning_rate = 1e-5
epochs = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# 이미지 id, 라벨 저장
image_ids = os.listdir(data_path)
image_labels = []
for json_path in json_paths:
    an_meta = json.load(open(json_path))['annotations']
    json_crop = (an_meta['crop'])
    json_disease = (an_meta['disease'])
    json_risk = (an_meta['risk'])
    label = f'{json_crop}_{json_disease}_{json_risk}'
    image_labels.append(label_encoder[label])

min_label = min(list(Counter(image_labels).values()))
max_label = max(list(Counter(image_labels).values()))


# 샘플링에 사용할 딕셔너리 세팅
image_set={}
for setting in list(set(image_labels)):
    image_set[setting]=[]
min_label = min(list(Counter(image_labels).values()))
# 딕셔너리 입력(labe : id)
for index in range(len(image_labels)):
    image_set[image_labels[index]].append(image_ids[index])


# f1 score
def accuracy_function(real, pred):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score


# Train step
def train_step(batch_item, training):
    image = batch_item[0].to(device)
    label = batch_item[1].to(device)
    feature = batch_item[2].to(device)
    if training is True:
        model.train()
        optimizer.zero_grad()

        output = model(image,feature)
        loss = loss_func(output, label)

        loss.backward()
        optimizer.step()
        accuarcy = accuracy_function(label, output)
        return loss, accuarcy

    else:
        model.eval()
        with torch.no_grad():
            output = model(image,feature)
            loss = loss_func(output, label)
        accuarcy = accuracy_function(label, output)
        return loss, accuarcy


# Network
model = Network(len(label_encoder))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

loss_plot, val_loss_plot = [], []
metric_plot, val_metric_plot = [], []


# Training
for epoch in range(epochs):
    for idx in range(int(max_label/min_label)):

        sampled_ids, sampled_labels = [],[]
        # 최소 갯수만큼 각 레이블에서 샘플링
        for label in list(set(image_labels)):
            sample_id = random.sample(image_set[label], min_label)  # 순환 중인 label에서 min_label의 갯수만큼 샘플링 (id 집단)
            sampled_ids.extend(sample_id)
            sampled_labels.extend([label for i in range(min_label)])

        sampled_dirs=[]
        for image_id in sampled_ids:
            sampled_dirs.append(f'{data_path}{image_id}/{image_id}.jpg')

        train_dirs, val_dirs, train_labels, val_labels = train_test_split(
            sampled_dirs, sampled_labels, test_size=0.3, random_state=42)

        train_dataset = CustomDataset(train_dirs, train_labels)
        val_dataset = CustomDataset(val_dirs, val_labels)

        train_dataloader = DataLoader(train_dataset, batch_size, drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size, drop_last=False)

        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0

        # train step
        training = True
        tqdm_dataset = tqdm(enumerate(train_dataloader))
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(batch_item, training)
            total_loss += batch_loss
            total_acc += batch_acc

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Sample Number' : f'{str(idx+1)}/{str(int(max_label / min_label))}',
                'Mean Loss': '{:06f}.'.format(total_loss / (batch + 1)),
                'Mean F1 Score': '{:06f}'.format(total_acc / (batch + 1))})

        # plot data
        loss_plot.append(total_loss.cpu().detach().numpy() / (batch + 1))
        metric_plot.append(total_acc / (batch + 1))
        # validatoin step
        tqdm_dataset = tqdm(enumerate(val_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(batch_item, training)
            total_val_loss += batch_loss
            total_val_acc += batch_acc

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Sample Number': f'{str(idx + 1)}/{str(int(max_label / min_label))}',
                'Mean Val Loss': '{:06f}.'.format(total_val_loss / (batch + 1)),
                'Mean Val F1 Score': '{:06f}'.format(total_val_acc / (batch + 1))})
        # plot data
        val_loss_plot.append(total_val_loss.cpu().detach().numpy() / (batch + 1))
        val_metric_plot.append(total_val_acc / (batch + 1))

        if np.max(val_metric_plot) == val_metric_plot[-1]:
           torch.save(model.state_dict(), './results/best_model_v5.pth')


plt.figure(figsize=(10,7))
plt.grid()
plt.plot(loss_plot, label='train_loss')
plt.plot(val_loss_plot, label='val_loss')
plt.xlabel('epoch+sample')
plt.ylabel('loss')
plt.title("Loss", fontsize=25)
plt.legend()
plt.savefig('./results/train_v5_loss_graph.jpg')
plt.show()

plt.figure(figsize=(10,7))
plt.grid()
plt.plot(metric_plot, label='train_loss')
plt.plot(val_metric_plot, label='val_loss')
plt.xlabel('epoch+sample')
plt.ylabel('accuracy')
plt.title("Accuracy", fontsize=25)
plt.legend()
plt.savefig('./results/train_v5_acc_graph.jpg')
plt.show()