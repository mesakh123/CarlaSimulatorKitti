## CarlaSimulatorKitti

Pygame CARLA Simulator to produce KITTI 2D/3D object detection

Source Code : https://github.com/mmmmaomao/DataGenerator

**Folder Format**

```
|-- dataset
    |-- training
    |   |-- calib/ # camera and lidar coeff
    |   |-- image/ # RGB image
    |   |-- labels/ # KITTI format image information
    |   |-- velodyne/ # 激光雷达的测量数据
    |   |-- train.txt
    |   |-- trainval.txt
    |   |-- val.txt

```

**KITTI label format**

```
 Values    Name      Description
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
Two types of labels:

1.  Actors : Car, Pedestrian
2.  Environment : None，Buildings，Fences，Other，Pedestrians，Poles，RoadLine，Roads，Sidewalks，TrafficSigns，Vegetation，Vehicles，Walls，Sky，Ground，Bridge，RailTrack，GuardRail，TrafficLight，Static，Dynamic，Water，Terrain

**Usage**

Carla Version：carla 0.9.12

Collecting Data

```
python3 generator.py
```

Only show pygame

```
python3 inference.py --loop
```

Collecting data and show Pygame

```
python3 inspector.py --loop
```

To enable predict on local model (put model on models/yolox_s.onnx)

```
python3 inference.py --loop --predict

or

python3 inspector.py --loop --predict

```

To enable predict (remote API, i.e 'http://<model_host>:<model_port>/predict')

```

python3 inference.py  --loop  --predict --model-host=0.0.0.0 --model-port=7777

```

**CarlaUtils**

SynchronyModel.py，Build client，setup server，generate actors，Generate data from server

data_utils.py， functions of coordinate transform、generate label

data_descriptor.py, KITTI format descripter

DataSave.py， object class to generate and save data folder

export_utils，saving data utils

image_converter.py, image format converter

visual_utils，visualization tools

### On Progress

- [x] Add Pygame
- [x] Enable to write file while running Pygame
- [x] Add YOLOX model option , local model or remote model
- [x] Improve writing file performance
- [ ] Enable manual control
- [ ] Others...
