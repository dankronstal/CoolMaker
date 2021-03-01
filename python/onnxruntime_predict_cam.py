# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import os
import sys
import onnxruntime
import onnx
import numpy as np
from PIL import Image, ImageDraw
from object_detection import ObjectDetection
import tempfile
from cv2 import *

MODEL_FILENAME = '..\\model.onnx'
LABELS_FILENAME = '..\\labels.txt'

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, os.path.basename(MODEL_FILENAME))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)

def main(image_filename):
    cam = VideoCapture(0)  #set the port of the camera as before
    
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = ONNXRuntimeObjectDetection(MODEL_FILENAME, labels)

    while True:
        retval, img = cam.read() #return a True bolean and and the image if all go right
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        predictions = od_model.predict_image(im_pil)

        opencvImage = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        #fx = 1000
        #fy = 750
        fx, fy = im_pil.size
        for k in predictions:
            if k['probability'] > 0.85:
                print(k['probability'])
                tl = (int(k['boundingBox']['left']*fx),int(k['boundingBox']['top']*fy))
                br = (int(k['boundingBox']['left']*fx+k['boundingBox']['width']*fx),int(k['boundingBox']['top']*fy+k['boundingBox']['height']*fy))

                opencvImage = cv2.rectangle(opencvImage,tl,br,(255, 0, 0),2)

        cv2.imshow('output', opencvImage)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows() # destroys the window showing image.
            cam.release() #Closes video file or capturing device.
            break
    
main('') #omitting param, since camera is input
