#!/usr/bin/env python3

import cv2
import sys
import os
import sys
import time

import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin

from multiprocessing import Process, Queue
import multiprocessing
import threading
import queue

infered_images = 0

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


def async_infer_worker(exe_net, request_number, image_queue, res_queue, input_blob, out_blob):
    global start_time
    global infered_images
    
    current_request_ids = range(request_number)
    next_request_ids = range(request_number, request_number * 2)
    done = False
    last_batch = -1
    while True:
        buffers = []
        for i in range(request_number):
            b = image_queue.get()
            if type(b) != np.ndarray:
                buffers.append(None)
                done = True
                break
            else:
                buffers.append(b)
        for _request_id in current_request_ids:
            if _request_id >=  request_number:
                if type(buffers[_request_id - request_number]) == np.ndarray:
                    exe_net.start_async(request_id=_request_id, inputs={input_blob: buffers[_request_id - request_number]})
                else:
                    #print("image at index " + str(_request_id - request_number) + " is none." )
                    last_batch = _request_id - request_number
                    break
            else:
                if type(buffers[_request_id]) == np.ndarray:
                    exe_net.start_async(request_id=_request_id, inputs={input_blob: buffers[_request_id]})
                else:
                    #print("image at index " + str(_request_id) + " is none." )
                    last_batch = _request_id
                    break
                    
        for _request_id in next_request_ids:
            if exe_net.requests[_request_id].wait(-1) == 0:
                res = exe_net.requests[_request_id].outputs[out_blob][0]
                res = np.array(res)
                res = np.transpose(res, (1,2,0))
                # print(res.shape)
                # cv2.imshow('Remaiaj', res)
                res = np.array(res/255).clip(0,1)

                res_queue.put(res)
                infered_images = infered_images + 1
                #print("infer result: label:%f confidence:%f left:%f top:%f right:%f bottom:%f" %(res[0][0][0][1], res[0][0][0][2], res[0][0][0][3], res[0][0][0][4], res[0][0][0][5], res[0][0][0][6]))
                duration = time.time() - start_time
                print("inferred images: " + str(infered_images) + ", average fps: " + str(infered_images/duration) +"\r", end = '', flush = False)

        current_request_ids, next_request_ids = next_request_ids, current_request_ids
        
        for i in range(len(buffers)):
            image_queue.task_done()
            
        if done:
            break

    # 'last_batch' more inference results remain to check
    buffer_index = 0
    for _request_id in next_request_ids:
        if(buffer_index >= last_batch):
            break
        buffer_index = buffer_index + 1
        if exe_net.requests[_request_id].wait(-1) == 0:
            res = exe_net.requests[_request_id].outputs[out_blob]
            infered_images = infered_images + 1
            #print("infer result: label:%f confidence:%f left:%f top:%f right:%f bottom:%f" %(res[0][0][0][1], res[0][0][0][2], res[0][0][0][3], res[0][0][0][4], res[0][0][0][5], res[0][0][0][6]))
            duration = time.time() - start_time
            print("inferred images: " + str(infered_images) + ", average fps: " + str(infered_images/duration) +"\r", end = '', flush = False)

# for test purpose only
image_number = 200

def preprocess_worker(cap, image_queue, ncs_number, n, c, h, w):
    global image_number_per_ncs
    
    for i in range(1, 1 + image_number):
        image = cap.read()
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        image = image * 255
        image_queue.put(image)
    # WTF
    # for i in range(ncs_number):
    #    image_queue.put(None)

start_time = -1

# ./async_api_multi-processes_multi-requests_multi-ncs.py <ncs number> <request number>

def load_model(root_name):
    model = root_name + ".xml"
    weights = root_name + ".bin"
    plugin = IEPlugin(device="MYRIAD") 
    net = IENetwork(model=model, weights=weights) 
    # exec_net = plugin.load(network=net)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    return net, plugin, input_blob, out_blob

def get_model_filepaths(model_dir):
    filepaths = set()
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            filepath = os.path.join(root, f)
            if ("bin" in f) or ("xml" in f):
                root_filepath = filepath.split('.')[1][1:]
                filepaths.add(root_filepath)
            else:
                print(f)

    filepaths = list(filepaths)
    print(filepaths)
    return filepaths

def main():
    global start_time
    cap = VideoCapture(0)
    # specify ncs number in argv
    ncs_number = int(sys.argv[1])
    # specify simutaneous request number in argv
    request_number = 1
    
    image_queue = queue.Queue(maxsize=ncs_number*request_number*3)
    res_queue = queue.Queue(maxsize=ncs_number*request_number*3)

    model_dir = sys.argv[2]
    model_filepaths = get_model_filepaths(model_dir)
    
    net, plugin, input_blob, out_blob = load_model(model_filepaths[0])
    n, c, h, w = net.inputs[input_blob].shape
    
    exec_nets = []
    for i in range(ncs_number):
        exec_net = plugin.load(network=net, num_requests=request_number*2)
        exec_nets.append(exec_net)

    start_time = time.time()

    preprocess_thread = threading.Thread(target=preprocess_worker, args=(cap, image_queue, ncs_number, n, c, h, w), daemon=True)
    preprocess_thread.start()
    
    infer_threads = [] 
    for f in range(ncs_number):
        _worker = threading.Thread(target=async_infer_worker, args=(exec_nets[f], request_number, image_queue, res_queue, input_blob, out_blob))
        _worker.start()
        infer_threads.append(_worker)

    print("Got here")
    if threading.current_thread() is threading.main_thread():
        # print("Main thread")
        while True:
            if res_queue.empty():
                continue
            # print("Not empty")
            img = res_queue.get()
            cv2.startWindowThread()
            cv2.namedWindow("preview")
            img = cv2.resize(img, (300,300))
            cv2.imshow("preview", img)


    preprocess_thread.join()
    for _worker in infer_threads:
        _worker.join()
    
    print()
    
    del exec_net
    del net
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)
