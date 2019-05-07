import threading
import binascii
from time import sleep
from utils import base64_to_pil_image, pil_image_to_base64
import sys
import os
from PIL import Image
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

class Camera(object):
    def __init__(self):
        self.to_process = []
        self.to_output = []

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()
    
    def predict(self, image_data, sess):
        predictions = sess.run(self.softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        max_score = 0.0
        res = ''
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score > max_score:
                max_score = score
                res = human_string
        return res, max_score

    def keep_processing(self):

        # Unpersists graph from file
        with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            self.softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            c = 0
            res, score = '', 0.0
            i = 0
            mem = ''
            consecutive = 0
            sequence = ''

            while True:
                if self.to_process:
                    input_str = self.to_process.pop(0)
                    # convert it to a pil image
                    img = base64_to_pil_image(input_str)
                    img = np.array(img)
                    
                    img = cv2.flip(img, 1)

                    x1, y1, x2, y2 = 100, 100, 300, 300
                    img_cropped = img[y1:y2, x1:x2]

                    c += 1
                    image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
                    
                    if i == 4:
                        res_tmp, score = self.predict(image_data, sess)
                        res = res_tmp
                        i = 0
                        if mem == res:
                            consecutive += 1
                        else:
                            consecutive = 0
                        if consecutive == 2 and res not in ['nothing']:
                            if res == 'space':
                                sequence += ' '
                            elif res == 'del':
                                sequence = sequence[:-1]
                            else:
                                sequence += res
                            consecutive = 0
                    i += 1
                    cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
                    cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                    mem = res
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.imshow("img", img)
                    img_sequence = np.zeros((200,1200,3), np.uint8)
                    cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    
                    frame = Image.fromarray(img_sequence)

                    output_str = pil_image_to_base64(frame)

                    # convert eh base64 string in ascii to base64 string in _bytes_
                    self.to_output.append(binascii.a2b_base64(output_str))
                sleep(0.01)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)