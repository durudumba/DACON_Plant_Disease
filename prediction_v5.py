import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch

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


chunk = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# test data load
test_data = './data/test/'
sub = pd.read_csv('./data/sample_submission.csv')
test_image_ids = sub['image']
test_image_dirs = []

for image_id in test_image_ids:
    test_image_dirs.append(f'{test_data}{image_id}/{image_id}.jpg')

test_dataset = CustomDataset(test_image_dirs, sub['label'])
test_dataloader = DataLoader(test_dataset, chunk, shuffle=False, drop_last=False)

# Prediction
def predict(dataset):
    model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    results = []
    for batch, batch_item in tqdm_dataset:
        image = batch_item[0].to(device)
        feature = batch_item[2].to(device)

        with torch.no_grad():
            output = model(image,feature)

        output = torch.argmax(output, dim=1).view(-1, 1).to(dtype=torch.int8)
        results.extend(output)

    return results

# Network parameter load
model = Network(len(label_encoder))
model.load_state_dict(torch.load('./results/best_model_v5.pth', map_location=device))
model.to(device)

preds = predict(test_dataloader)
preds = np.array([label_decoder[int(val)] for val in preds])

sub['label'] = preds
sub.to_csv('./results/submission_v5.csv', index=False)