# CoolMaker
This project was a quick test to use an AI model exported from Microsoft's [Custom Vision service](https://www.customvision.ai/). Small adaptation of exported code content to use webcam instead of file as input, and to show a bounding box around the detected object using cv2.

# Required Environment
In order to run the code in this repo the following environment and libraries must be available:

- Python 3.8
- [NumPy](https://numpy.org/)
- [Pillow](https://github.com/python-pillow/Pillow/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [ONNX Runtime](https://aka.ms/onnxruntime/) 

# Running Output
![Video of running code](Assets/coolmaker.gif)

# Notes
As visible in the video, the blue bounding box correctly identifies the item. Mostly. Numbers in the terminal output represent inference probability per frame, with a cutoff at 85% (no box or number if the probability is less than 85%). It's interesting to see that in my attached model the rear of the object was not well captured in the model and so when rotated to present that view the probability drops below the threshold. 

Export your own ONNX model and run this code to try it for yourself! (or if your daughter also has a CoolMaker then I guess you could try that...)

# Usage
>python.exe <repo>\python\onnxruntime_predict_cam.py