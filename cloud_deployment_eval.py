import os
import io
import neptune
import torch
import requests
import json
import time
import numpy as np
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from time import perf_counter_ns
import nncompression.utils as nnu
from nncompression.experiments import utils as exu
from nncompression.models.pytorch.utils import get_imagenet_val_loader
from nncompression.utils import IMAGENET_LABELS, DEVICE

BASE_URL = 'http://ml-inference.northeurope.cloudapp.azure.com/'
INFERENCE_URL = 'inference'
AVAIL_MODELS_URL = 'avail_models'
def SET_MODEL_URL(model): return f'set_model/{model}'


r = requests.get(BASE_URL+AVAIL_MODELS_URL)
AVAILABLE_MODELS = r.json()['available_models']

for model in AVAILABLE_MODELS:

    r = requests.post(BASE_URL+f'set_model/{model}', json={'api_key': 'password'})
    model_info = r.json()

    neptune.init('davidturner94/deployment-eval')

    PARAMS = model_info

    experiment = neptune.create_experiment(name=model, params=PARAMS)

    val_loader = get_imagenet_val_loader('data/imagenet', batch_size=1)

    batch_time = exu.AverageMeter('Time', ':6.3f')
    inference_time = exu.AverageMeter('inference_time', ':6.3f')
    latency_out = exu.AverageMeter('Latency_out', ':6.3f')
    latency_back = exu.AverageMeter('Latency_back', ':6.3f')
    power_usage = exu.AverageMeter('Power_in_watts', ':.1f')
    gpu_utilization = exu.AverageMeter('GPU_utilization', ':.1f')
    top1 = exu.AverageMeter('Acc@1', ':6.2f')
    top5 = exu.AverageMeter('Acc@5', ':6.2f')

    progress = exu.ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5, inference_time, latency_out, latency_back, power_usage, ],
        prefix='Test: ')

    print_freq = 5000
    request_type = 'pil'

    class_correct = list({'top1': 0., 'top5': 0., 'total': 0.} for _ in range(len(IMAGENET_LABELS)))

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(val_loader):

            im = transforms.ToPILImage()(images[0])

            # convert pil image to bytes
            with BytesIO() as output:
                im.save(output, 'JPEG')
                data = output.getvalue()

            payload = {
                'files': (
                    '1.jpeg',
                    data,
                    'image/jpeg'
                )
            }
            time_request_sent = time.time()
            r = requests.post(BASE_URL+INFERENCE_URL, files=payload)
            time_response_recieved = time.time()
            r_json = r.json()

            outputs = torch.tensor(
                r_json['result']['prediction_raw']).unsqueeze(0)

            outputs, labels = outputs.to(DEVICE), labels.to(DEVICE)

            acc1, acc5 = exu.accuracy(outputs, labels, topk=(1, 5))

    #         print(torch.nn.functional.softmax(outputs).topk(5))

            class_correct[labels[0].item()]['total'] += 1

            if acc1[0] == 100.:
                class_correct[labels[0].item()]['top1'] += 1

            if acc5[0] == 100.:
                class_correct[labels[0].item()]['top5'] += 1

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            inference_time.update(r_json['result']['inference_time_ms'])

            power_usage.update(r_json['meta']['cuda_info']['gpu'][0]
                               ['power_readings']['power_draw'], images.size(0))
            gpu_utilization.update(r_json['meta']['cuda_info']['gpu']
                                   [0]['utilization']['gpu_util'], images.size(0))

            latency_out.update(r_json['meta']['time_request_recieved']-time_request_sent)
            latency_back.update(time_response_recieved-r_json['meta']['time_response_sent'])

            if i % print_freq == 0:
                progress.display(i)

            neptune.log_metric('batch_time', batch_time.val)
            neptune.log_metric('inference_time', inference_time.val)
            neptune.log_metric('latency_out', latency_out.val)
            neptune.log_metric('latency_back', latency_back.val)
            neptune.log_metric('top1_accuracy_avg', top1.avg)
            neptune.log_metric('top1_accuracy_raw', top1.val)
            neptune.log_metric('top5_accuracy_avg', top5.avg)
            neptune.log_metric('top5_accuracy_raw', top5.val)
            neptune.log_metric('gpu_power_w', power_usage.val)
            neptune.log_metric('gpu_util', gpu_utilization.val)

        results = {
            r_json['meta']['model']['name']: {
                'batch_time': {
                    'avg': batch_time.avg,
                    'max': batch_time.max,
                    'min': batch_time.min
                },
                'inference_time': {
                    'avg': inference_time.avg,
                    'max': inference_time.max,
                    'min': inference_time.min
                },
                'latency_out': {
                    'avg': latency_out.avg,
                    'max': latency_out.max,
                    'min': latency_out.min
                },
                'latency_back': {
                    'avg': latency_back.avg,
                    'max': latency_back.max,
                    'min': latency_back.min
                },
                'top1_accuracy': {
                    'avg': top1.avg.item(),
                },
                'top5_accuracy': {
                    'avg': top5.avg.item(),
                },
                'gpu_power_usage': {
                    'avg': power_usage.avg,
                    'max': power_usage.max,
                    'min': power_usage.min
                },
                'gpu_util': {
                    'avg': gpu_utilization.avg,
                    'max': gpu_utilization.max,
                    'min': gpu_utilization.min
                },
            }
        }
        neptune.log_text('results', json.dumps(results))
        class_accuracy = '\n'.join([json.dumps(dict(x, **{'top1_accuracy': 100*x['top1']/x['total'], 'top5_accuracy': 100*x['top5']/x['total']}, **{
                                   "class": IMAGENET_LABELS[i]})) for i, x in enumerate(class_correct)])
        neptune.log_text('class_accuracy', class_accuracy)
    neptune.stop()
