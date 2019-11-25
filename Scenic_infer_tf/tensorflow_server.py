import os
import uuid
import numpy as np
import cv2 as cv
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 创建执行器
inference_pb = "./frozen_inference_graph.pb"
graph_txt = "./graph.pbtxt"
net = cv.dnn.readNetFromTensorflow(inference_pb, graph_txt)


@app.route('/infer', methods=['POST'])
def infer():
    f = request.files['img']

    # 保存图片
    save_father_path = 'images'
    img_path = os.path.join(save_father_path, str(uuid.uuid1()) + '.' + secure_filename(f.filename).split('.')[-1])
    if not os.path.exists(save_father_path):
        os.makedirs(save_father_path)
    f.save(img_path)

    image = cv.imread(img_path)
    h, w = image.shape[:2]
    im_tensor = cv.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
    net.setInput(im_tensor)

    cvOut = net.forward()
    print(cvOut.shape)
    goal = "0"
    num = 1000
    what = "无"
    for detect in cvOut[0, 0, :, :]:
        score = detect[2]
        if score > 0.7:
            num = detect[1]  # 类别
            if num == 1.0:
                what = "普贤塔"
            if num == 0.0:
                what = "象山岩"
        goal = "{:.2f}%".format(score * 100)
    # 打印和返回预测结果
    r = '{"label":%d, "name":"%s", "possibility":%s}' % (num, what, goal)
    # r = '%s' % what
    print(r)
    return r


if __name__ == '__main__':
    # 启动服务，并指定端口号
    # app.run(port=80)
    app.run(host="0.0.0.0", port=8989)
