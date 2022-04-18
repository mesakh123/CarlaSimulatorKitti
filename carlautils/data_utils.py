import sys

sys.path.append("..")  # Adds higher directory to python modules path.
import numpy as np
from numpy.linalg import inv

from config import cfg_from_yaml_file
from .data_descriptor import KittiDescriptor, CarlaDescriptor
from .image_converter import depth_to_array, to_rgb_array
import math
from utils.visual_utils import draw_3d_bounding_box

sys.path.append(
    "/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg"
)

import carla

cfg = cfg_from_yaml_file("configs.yaml")

MAX_RENDER_DEPTH_IN_METERS = cfg["FILTER_CONFIG"]["MAX_RENDER_DEPTH_IN_METERS"]
MIN_VISIBLE_VERTICES_FOR_RENDER = cfg["FILTER_CONFIG"][
    "MIN_VISIBLE_VERTICES_FOR_RENDER"
]
MAX_OUT_VERTICES_FOR_RENDER = cfg["FILTER_CONFIG"]["MAX_OUT_VERTICES_FOR_RENDER"]
WINDOW_WIDTH = cfg["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_x"]
WINDOW_HEIGHT = cfg["SENSOR_CONFIG"]["DEPTH_RGB"]["ATTRIBUTE"]["image_size_y"]

filter_list = {"TrafficLight": 70}
filter_list_area = {"RoadLine": WINDOW_HEIGHT * WINDOW_WIDTH / 4}


def obj_type(obj):
    allowed = ["poles", "fences"]
    string = ""
    if isinstance(obj, carla.EnvironmentObject):
        string = str(obj.type)
    else:
        if obj.type_id.find("walker") != -1:
            string = "Pedestrian"
        if obj.type_id.find("vehicle") != -1:
            string = "Vehicles"
        if obj.type_id.find("traffic_light") != -1:
            string = "TrafficLight"
        if obj.type_id.find("speed_limit") != -1:
            string = "TrafficSigns"
        string = "None"
    if string[-1] == "s" and string.lower() not in allowed:
        string = string[:-1]
    if string.lower() == "none":
        string = "None"
    return string


def get_relative_rotation_y(agent_rotation, obj_rotation):
    """返回actor和camera在rotation yaw的相对角度"""

    rot_agent = agent_rotation.yaw
    rot_car = obj_rotation.yaw
    return degrees_to_radians(rot_agent - rot_car)


def bbox_2d_from_agent(intrinsic_mat, extrinsic_mat, obj_bbox, obj_transform, obj_tp):
    bbox = vertices_from_extension2(obj_bbox.extent)
    bbox_transform = carla.Transform(obj_bbox.location, obj_transform.rotation)

    if obj_tp == 1:
        bbox = transform_points(bbox_transform, bbox)
        bbox = transform_points(obj_transform, bbox)

    else:
        if obj_tp != 3:
            bbox = transform_points(bbox_transform, bbox)
        else:
            bbox = vertices_from_extension(obj_bbox.extent)
            box_location = carla.Location(
                obj_bbox.location.x - obj_transform.location.x,
                obj_bbox.location.y - obj_transform.location.y,
                obj_bbox.location.z - obj_transform.location.z,
            )
            box_rotation = obj_bbox.rotation
            bbox_transform = carla.Transform(box_location, box_rotation)
            bbox = transform_points(bbox_transform, bbox)
            bbox = transform_points(obj_transform, bbox)

    vertices_pos2d = vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat)
    return vertices_pos2d


def bbox_2d_from_agent_ori(
    intrinsic_mat, extrinsic_mat, obj_bbox, obj_transform, obj_tp
):
    bbox = vertices_from_extension(obj_bbox.extent)
    if obj_tp == 1:
        bbox_transform = carla.Transform(obj_bbox.location, obj_bbox.rotation)
        bbox = transform_points(bbox_transform, bbox)
    else:
        box_location = carla.Location(
            obj_bbox.location.x - obj_transform.location.x,
            obj_bbox.location.y - obj_transform.location.y,
            obj_bbox.location.z - obj_transform.location.z,
        )
        box_rotation = obj_bbox.rotation
        bbox_transform = carla.Transform(box_location, box_rotation)
        bbox = transform_points(bbox_transform, bbox)
    # 获取bbox在世界坐标系下的点的坐标
    bbox = transform_points(obj_transform, bbox)
    # 将世界坐标系下的bbox八个点转换到二维图片中
    vertices_pos2d = vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat)
    return vertices_pos2d


def vertices_from_extension(ext):
    """以自身为原点的八个点的坐标"""
    return np.array(
        [
            [ext.x, ext.y, ext.z],  # Top left front
            [-ext.x, ext.y, ext.z],  # Top left back
            [ext.x, -ext.y, ext.z],  # Top right front
            [-ext.x, -ext.y, ext.z],  # Top right back
            [ext.x, ext.y, -ext.z],  # Bottom left front
            [-ext.x, ext.y, -ext.z],  # Bottom left back
            [ext.x, -ext.y, -ext.z],  # Bottom right front
            [-ext.x, -ext.y, -ext.z],  # Bottom right back
        ]
    )


def vertices_from_extension2(extent):
    cords = np.zeros((8, 3))
    cords[0, :] = np.array([extent.x, extent.y, -extent.z])  # Top left front
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z])  # Top left back
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z])  # Top right front
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z])  # Top right back
    cords[4, :] = np.array([extent.x, extent.y, extent.z])  # Bottom left front
    cords[5, :] = np.array([-extent.x, extent.y, extent.z])  # Bottom left back
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z])  # Bottom right front
    cords[7, :] = np.array([extent.x, -extent.y, extent.z])  # Bottom right back
    return cords


def transform_points(transform, points):
    points = points.transpose()
    # [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]  (4,8)
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    points = np.mat(transform.get_matrix()) * points
    return points[0:3].transpose()


def vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat):
    """
    Accepts a bbox which is a list of 3d world coordinates and returns a list
    of the 2d pixel coordinates of each vertex.
    This is represented as a tuple (y, x, d) where y and x are the 2d pixel coordinates
    while d is the depth. The depth can be used for filtering visible vertices.
    """
    vertices_pos2d = []
    for vertex in bbox:
        # Get vector from world coor system
        pos_vector = vertex_to_world_vector(vertex)
        # Camera coordinates
        transformed_3d_pos = proj_to_camera(pos_vector, extrinsic_mat)
        # 2d pixel coordinates
        pos2d = proj_to_2d(transformed_3d_pos, intrinsic_mat)
        # Actual rendered depth
        vertex_depth = pos2d[2]
        # Coordinate of points on image
        x_2d, y_2d = pos2d[0], pos2d[1]
        vertices_pos2d.append((y_2d, x_2d, vertex_depth))
    return vertices_pos2d


def vertex_to_world_vector(vertex):
    """以carla世界向量（X，Y，Z，1）返回顶点的坐标 （4,1）"""
    return np.array(
        [
            [vertex[0, 0]],  # [[X,
            [vertex[0, 1]],  # Y,
            [vertex[0, 2]],  # Z,
            [1.0],  # 1.0]]
        ]
    )


def calculate_occlusion_stats(vertices_pos2d, depth_image):
    """
    Filtering 8 points of bbox visible vertices
    """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0
    for y_2d, x_2d, vertex_depth in vertices_pos2d:
        # if the point is in front of the camera but not too far away
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas(
            (y_2d, x_2d)
        ):
            is_occluded = point_is_occluded((y_2d, x_2d), vertex_depth, depth_image)
            if not is_occluded:
                num_visible_vertices += 1
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def point_in_canvas(pos):
    if (
        (pos[0] >= 0)
        and (pos[0] < WINDOW_HEIGHT)
        and (pos[1] >= 0)
        and (pos[1] < WINDOW_WIDTH)
    ):
        return True
    return False


def point_is_occluded(point, vertex_depth, depth_image):
    y, x = map(int, point)
    from itertools import product

    neigbours = product((1, -1), repeat=2)
    is_occluded = []
    for dy, dx in neigbours:
        if point_in_canvas((dy + y, dx + x)):
            # 判断点到图像的距离是否大于深对应深度图像的深度值
            if depth_image[y + dy, x + dx] < vertex_depth:
                is_occluded.append(True)
            else:
                is_occluded.append(False)
    # 当四个邻居点都大于深度图像值时，点被遮挡。返回true
    return all(is_occluded)


def midpoint_from_agent_location(location, extrinsic_mat):
    """
    Transform agent world coordinate system center point into camera coordinate system
    Calculate the midpoint of the bottom chassis
    This is used since kitti treats this point as the location of the car

    """

    midpoint_vector = np.array(
        [[location.x], [location.y], [location.z], [1.0]]  # [[X,  # Y,  # Z,  # 1.0]]
    )
    transformed_3d_midpoint = proj_to_camera(midpoint_vector, extrinsic_mat)
    return transformed_3d_midpoint


def camera_intrinsic(width, height):
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    f = width / (2.0 * math.tan(90.0 * math.pi / 360.0))
    k[0, 0] = k[1, 1] = f
    return k


def proj_to_camera(pos_vector, extrinsic_mat):
    """
    Convert world coordinate system point into camera coordinate system
    """
    # inv
    transformed_3d_pos = np.dot(inv(extrinsic_mat), pos_vector)
    return transformed_3d_pos


def proj_to_2d(camera_pos_vector, intrinsic_mat):

    cords_x_y_z = camera_pos_vector[:3, :]
    cords_y_minus_z_x = np.concatenate(
        [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]]
    )
    pos2d = np.dot(intrinsic_mat, cords_y_minus_z_x)
    pos2d = np.array([pos2d[0] / pos2d[2], pos2d[1] / pos2d[2], pos2d[2]])
    return pos2d


def filter_by_distance(data_dict, dis):
    environment_objects = data_dict["environment_objects"]
    actors = data_dict["actors"]
    for agent, _ in data_dict["agents_data"].items():
        data_dict["environment_objects"] = [
            obj
            for obj in environment_objects
            if distance_between_locations(obj.transform.location, agent.get_location())
            < dis
        ]
        data_dict["actors"] = [
            act
            for act in actors
            if distance_between_locations(act.get_location(), agent.get_location())
            < dis
        ]


def distance_between_locations(location1, location2):
    return math.sqrt(
        pow(location1.x - location2.x, 2) + pow(location1.y - location2.y, 2)
    )


def calc_projected_2d_bbox(vertices_pos2d):
    """
    Takes in all vertices in pixel projection and calculates min and max of all x and y coordinates.
    Returns left top, right bottom pixel coordinates for the 2d bounding box as a list of four values.
    Note that vertices_pos2d contains a list of (y_pos2d, x_pos2d) tuples, or None
    """
    legal_pos2d = list(filter(lambda x: x is not None, vertices_pos2d))
    y_coords, x_coords = [int(x[0][0]) for x in legal_pos2d], [
        int(x[1][0]) for x in legal_pos2d
    ]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    image_width = int(cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"])
    image_height = int(cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"])

    min_x, max_x = max(0, min_x), min(image_width, max_x)
    min_y, max_y = max(0, min_y), min(image_height, max_y)

    return [min_x, min_y, max_x, max_y]


def degrees_to_radians(degrees):
    return degrees * math.pi / 180


def custom_calc_projected_2d_bbox(vertices_pos2d, depth_image, object_type=0):
    """
    Takes in all vertices in pixel projection and calculates min and max of all x and y coordinates.
    Returns left top, right bottom pixel coordinates for the 2d bounding box as a list of four values.
    Note that vertices_pos2d contains a list of (y_pos2d, x_pos2d) tuples, or None
    """
    legal_pos2d = list(filter(lambda x: x is not None, vertices_pos2d))
    y_coords, x_coords = [int(x[0][0]) for x in legal_pos2d], [
        int(x[1][0]) for x in legal_pos2d
    ]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    image_width = int(cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"])
    image_height = int(cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"])

    min_x, max_x = max(0, min_x), min(image_width, max_x)
    min_y, max_y = max(0, min_y), min(image_height, max_y)

    return [min_x, min_y, max_x, max_y]


def legal_bbox(bbox, obj_tp):

    min_area = 40
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    if obj_tp in filter_list.keys():
        min_area = filter_list[obj_tp]

    is_min_area = True if area > min_area else False

    is_max_area = True
    if obj_tp in filter_list_area.keys():
        is_max_area = False if area >= filter_list_area[obj_tp] else True

    return is_min_area and is_max_area


def objects_filter(data):
    environment_objects = data["environment_objects"]
    agents_data = data["agents_data"]
    actors = data["actors"]
    actors = [
        x
        for x in actors
        if x.type_id.find("vehicle") != -1
        or x.type_id.find("walker") != -1
        or x.type_id.find("traffic") != -1
    ]
    for agent, dataDict in agents_data.items():
        intrinsic = dataDict["intrinsic"]
        extrinsic = dataDict["extrinsic"]
        sensors_data = dataDict["sensor_data"]
        kitti_datapoints = []
        carla_datapoints = []
        rgb_image = to_rgb_array(sensors_data[0])
        image = rgb_image.copy()
        depth_data = sensors_data[1]
        radar_datapoints = sensors_data[3]

        data["agents_data"][agent]["visible_environment_objects"] = []
        for obj in environment_objects:
            kitti_datapoint, carla_datapoint = is_visible_by_bbox(
                agent, obj, image, depth_data, intrinsic, extrinsic
            )
            if kitti_datapoint is not None:
                data["agents_data"][agent]["visible_environment_objects"].append(obj)
                kitti_datapoints.append(kitti_datapoint)
                carla_datapoints.append(carla_datapoint)

        data["agents_data"][agent]["visible_actors"] = []

        for act in actors:
            kitti_datapoint, carla_datapoint = is_visible_by_bbox(
                agent, act, image, depth_data, intrinsic, extrinsic
            )
            if kitti_datapoint is not None:
                data["agents_data"][agent]["visible_actors"].append(act)
                kitti_datapoints.append(kitti_datapoint)
                carla_datapoints.append(carla_datapoint)

        data["agents_data"][agent]["rgb_image"] = image
        data["agents_data"][agent]["kitti_datapoints"] = kitti_datapoints
        data["agents_data"][agent]["carla_datapoints"] = carla_datapoints
        data["agents_data"][agent]["radar_datapoints"] = radar_datapoints
    return data


def is_visible_by_bbox(agent, obj, rgb_image, depth_data, intrinsic, extrinsic):
    actor_type_list = [carla.Walker, carla.Vehicle, carla.WalkerAIController]
    object_type = 1 if type(obj) in actor_type_list else 0

    obj_tp = obj_type(obj)
    if isinstance(obj_tp, str) and (obj_tp == "TrafficLight"):
        object_type = 2

    obj_transform = (
        obj.transform
        if isinstance(obj, carla.EnvironmentObject)
        else obj.get_transform()
    )

    obj_bbox = obj.bounding_box
    vertices_pos2d = bbox_2d_from_agent(
        intrinsic, extrinsic, obj_bbox, obj_transform, object_type
    )

    depth_image = depth_to_array(depth_data)
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
        vertices_pos2d, depth_image
    )
    if (
        num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER
        and num_vertices_outside_camera < MAX_OUT_VERTICES_FOR_RENDER
    ):
        midpoint = midpoint_from_agent_location(obj_transform.location, extrinsic)
        # midpoint[:3] = np.identity(3) * midpoint[:3]

        # bbox_2d = calc_projected_2d_bbox(vertices_pos2d)

        bbox_2d = custom_calc_projected_2d_bbox(
            vertices_pos2d, depth_image, object_type
        )

        if not legal_bbox(bbox_2d, str(obj_tp)):
            return None, None

        rotation_y = (
            get_relative_rotation_y(
                agent.get_transform().rotation, obj_transform.rotation
            )
            % math.pi
        )

        ext = obj.bounding_box.extent
        truncated = num_vertices_outside_camera / 8
        if num_visible_vertices >= 6:
            occluded = 0
        elif num_visible_vertices >= 4:
            occluded = 1
        else:
            occluded = 2

        velocity = (
            "0 0 0"
            if isinstance(obj, carla.EnvironmentObject)
            else "{} {} {}".format(
                obj.get_velocity().x, obj.get_velocity().y, obj.get_velocity().z
            )
        )
        acceleration = (
            "0 0 0"
            if isinstance(obj, carla.EnvironmentObject)
            else "{} {} {}".format(
                obj.get_acceleration().x,
                obj.get_acceleration().y,
                obj.get_acceleration().z,
            )
        )
        angular_velocity = (
            "0 0 0"
            if isinstance(obj, carla.EnvironmentObject)
            else "{} {} {}".format(
                obj.get_angular_velocity().x,
                obj.get_angular_velocity().y,
                obj.get_angular_velocity().z,
            )
        )
        # draw_3d_bounding_box(rgb_image, vertices_pos2d)

        kitti_data = KittiDescriptor()
        kitti_data.set_truncated(truncated)
        kitti_data.set_occlusion(occluded)
        kitti_data.set_bbox(bbox_2d)
        kitti_data.set_3d_object_dimensions(ext)
        kitti_data.set_type(obj_tp)
        kitti_data.set_3d_object_location(midpoint)
        kitti_data.set_rotation_y(rotation_y)

        carla_data = CarlaDescriptor()
        carla_data.set_type(obj_tp)
        carla_data.set_velocity(velocity)
        carla_data.set_acceleration(acceleration)
        carla_data.set_angular_velocity(angular_velocity)
        return kitti_data, carla_data
    return None, None
