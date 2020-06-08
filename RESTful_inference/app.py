from flask import Flask, jsonify, request
from flask_redis import FlaskRedis
from celery import Celery
from time import perf_counter_ns
from context import nncompression
from nncompression import utils
import torchvision.models as models


app = Flask(__name__)


model = models.resnet18(pretrained=True)
model.eval()
model.to(utils.DEVICE)


@app.route("/")
def index():
    response = {
            "status": "All Good.",
            }
    return jsonify(response)


@app.route("/inference", methods=["POST"])
def inference():
    """
    Method for inference on a provided image

    This method takes a json request containing an image and the model to test
    and returns json data containing the predicted result and some meta data
    relating to inference time and confidence score.


    """

    image_bn = request.json['image']

    # convert image from binary to format model accepts

    # run inference
    perf_start = perf_counter_ns()

    # inference here
    out = model(image_bn)

    perf_end = perf_counter_ns()

    # return results as a json
    return jsonify({
        "result": {
            "prediction_label": "",
            "confidence": "",
            "top_5": "",
            "recall": "",
            "precision": "",
            "f1": "",
            "confusion_matrix": ""

        },
        "meta": {
            "performance_ms": (perf_end-perf_start) / 10**6,
            "device": "",
            "model": {
                "name": "",
                "size": "",
                "no_params": "",
                },
            "time_now": "", # This is to enable calculation of latency
            "power_draw": ""

        }
    })


if __name__ == "__main__":
    app.run(debug=True)
