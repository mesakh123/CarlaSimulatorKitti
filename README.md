## CarlaSimulatorKitti

Pygame CARLA Simulator to produce KITTI 2D/3D object detection

Source Code : https://github.com/mmmmaomao/DataGenerator

**Folder Format**

dataset |
training

      |\_\_ calib/ # camera and lidar coeff

      |\_\_ image/ # RGB image

      |\_\_ label/ # object 的标签

      |\_\_ velodyne/ # 激光雷达的测量数据

      |\_\_ train.txt

      |\_\_ trainval.txt

      |\_\_ val.txt

```
label：
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car','Pedestrian',
   					 'TrafficSigns', etc.
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
```

**label type**

label 标定的目标主要分为两类，第一类是我们自己生成的 actors(Car 和 Pedestrian)；第二类是地图中存在的环境目标(None，Buildings，Fences，Other，Pedestrians，Poles，RoadLines，Roads，Sidewalks，TrafficSigns，Vegetation，Vehicles，Walls，Sky，Ground，Bridge，RailTrack，GuardRail，TrafficLight，Static，Dynamic，Water，Terrain)

**Usage**

Carla 版本：carla 0.9.12

Only collecting Data and show Pygame

```
python3 generator.py
```

To enable predict on local model (put model on models/yolox_s.onnx)

```
python3 generator.py --predict
```

To enable predict (remote API, i.e 'http://<model_host>:<model_port>/predict')

```
python3 generator.py --predict --model-host=0.0.0.0 --model-port=7777
```

SynchronyModel.py，场景类，负责建立 client，设置 server，生成 actors，驱动 server 计算并获取数据

data_utils.py，包含点坐标转换、生成 label 等工具函数

data_descriptor.py, KITTI 格式的描述类

DataSave.py，数据保存类，生成保存数据路径，保存数据

export_utils，保存数据的工具函数

image_converter.py, 图片格式转换函数

visual_utils，可视化工具函数

### On Progress

- [x] Add Pygame
- [x] Enable to write file while running Pygame
- [x] Add YOLOX model option , local model or remote model
- [ ] Enable manual control
- [ ] Improve writing file performance
- [ ] Others...
