from openvino.inference_engine import IENetwork, IEPlugin
import random
import argparse
import collections
import os
from multiprocessing import Queue
import threading
import time

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.autograd import Variable


# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except:
                    # pass
                    continue
            self.q.put(frame)

    def read(self):
        return self.q.get()


filepaths = set()
for root, dirs, files in os.walk('.'):
    for f in files:
        filepath = os.path.join(root, f)
        if "bin" in f or "xml" in f:
            filepaths.add(filepath[2:].split('.')[0])

filepaths = list(filepaths)

def load_model(root_name):
    model = root_name + ".xml"
    weights = root_name + ".bin"
    plugin = IEPlugin(device="MYRIAD") 
    net = IENetwork(model=model, weights=weights) 
    exec_net = plugin.load(network=net)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    return exec_net, input_blob, out_blob

def itot(img, max_size=None):
    # Rescale the image
    if (max_size==None):
        itot_t = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    else:
        H, W, C = img.shape
        image_size = tuple([int((float(max_size) / max([H,W]))*x) for x in [H, W]])
        itot_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    # Convert image to tensor
    tensor = itot_t(img)

    # Add the batch_size dimension
    tensor = tensor.unsqueeze(dim=0)
    return tensor


print(filepaths)
if __name__ == "__main__":
    cap = VideoCapture(0)

    random_model = None
    prev_random = None
    exec_net, input_blob, out_blob = load_model(random_model)
    while True:
        # Prepare input frame
        # Stylize image
        img = cap.read()
        if img is None:
            continue
        # cv2.imshow('plm', img)
        if chr(cv2.waitKey(1) & 255) == 'q':
            break

        img = cv2.resize(img, (128, 128))
        img = np.array([img])
        img = np.transpose(img,(0,3,1,2))
        # 
        # print(img.shape)
        # plm nebunie
        img = img * 255
        res = exec_net.infer(inputs={input_blob: img})
        # print(res)
        res = res[out_blob][0]
        # plm nebunie
        res = np.array(res/255).clip(0,1)

        res = np.transpose(res, (1,2,0))
        # print(res.shape)

        # print(stylized_image.shape)
        if res is not None:
            res = cv2.resize(res, (300,300))
            cv2.imshow('Darius Omaj', res)
