pip3 install -r requirements.txt
python3 -m pip install --upgrade tensorrt
python3 -m pip install numpy
python3 -m pip install 'pycuda<2021.1'
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
# python3 setup.py develop ### yolox
# python setup.py install ### torch2trt
