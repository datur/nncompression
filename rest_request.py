import requests
from nncompression.models.pytorch.utils import get_imagenet_val_loader
from nncompression.utils import IMAGENET_LABELS
import os
import io
from io import BytesIO
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import nncompression.utils as nnutil
import base64
import json
from tqdm import tqdm

os.environ['NO_PROXY'] = '127.0.0.1'

val_loader = get_imagenet_val_loader('data/imagenet', batch_size=1)

correct = 0
total = 0
class_correct = list(0. for _ in range(len(IMAGENET_LABELS)))
class_total = list(0. for _ in range(len(IMAGENET_LABELS)))

for (img, lab), r in zip(tqdm(val_loader), range(10)):
    im = transforms.ToPILImage()(img[0])
    with BytesIO() as output:
        im.save(output, 'JPEG')
        data = output.getvalue()

    url = "http://127.0.0.1:5000/inference"
    headers = {'Content-Type': 'application/octet-stream'}
    r = requests.post(url, data=data, headers=headers)
    r_json = r.json()

    pred = torch.tensor(r_json['result']['predicted_class'])

    correct += (pred == lab).sum().item()
    total += 1

    label = lab[0]
    class_correct[label] += 1 if (pred == lab) else 0
    class_total[label] += 1

print([100 * class_correct[i] / class_total[i] for i in range(len(IMAGENET_LABELS))])
print(100 * correct / total)
# print(f"{lab[0].item()} {r_json['result']['predicted_class'][0]} \t {r_json['result']['inference_time_ms']}")
