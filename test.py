import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla
import logging
import math
import random
import queue
import numpy as np
from math import pi
from typing import List

import image_converter

""" OUTPUT FOLDER GENERATION """
PHASE = "training"
OUTPUT_FOLDER = os.path.join("_out", PHASE)
folders = ["calib", "image_2", "label_2", "velodyne", "planes"]

""" DATA GENERATION SETTINGS"""
GEN_DATA = True  # Whether or not to save training data
VISUALIZE_LIDAR = False
# How many frames to wait between each capture of screen, bounding boxes and lidar
STEPS_BETWEEN_RECORDINGS = 10
CLASSES_TO_LABEL = ["Vehicle", "Pedestrian"]
# Lidar can be saved in bin to comply to kitti, or the standard .ply format
LIDAR_DATA_FORMAT = "bin"
assert LIDAR_DATA_FORMAT in [
    "bin",
    "ply",
], "Lidar data format must be either bin or ply"
OCCLUDED_VERTEX_COLOR = (255, 0, 0)
VISIBLE_VERTEX_COLOR = (0, 255, 0)
# How many meters the car must drive before a new capture is triggered.
DISTANCE_SINCE_LAST_RECORDING = 10
# How many datapoints to record before resetting the scene.
NUM_RECORDINGS_BEFORE_RESET = 20
# How many frames to render before resetting the environment
# For example, the agent may be stuck
NUM_EMPTY_FRAMES_BEFORE_RESET = 100

""" CARLA SETTINGS """
CAMERA_HEIGHT_POS = 2.0
LIDAR_HEIGHT_POS = CAMERA_HEIGHT_POS
MIN_BBOX_AREA_IN_PX = 100


""" AGENT SETTINGS """
NUM_VEHICLES = 20
NUM_PEDESTRIANS = 10

""" RENDERING SETTINGS """
WINDOW_WIDTH = 1248
WINDOW_HEIGHT = 384
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180

WINDOW_WIDTH_HALF = WINDOW_WIDTH / 2
WINDOW_HEIGHT_HALF = WINDOW_HEIGHT / 2

MAX_RENDER_DEPTH_IN_METERS = 70  # Meters
MIN_VISIBLE_VERTICES_FOR_RENDER = 4

NUM_OF_WALKERS = 30
FILTERW = "walker.pedestrian.*"
FILTERV = "vehicle.*"
NUM_OF_VEHICLES = 20


def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)


for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)

""" DATA SAVE PATHS """
GROUNDPLANE_PATH = os.path.join(OUTPUT_FOLDER, "planes/{0:06}.txt")
LIDAR_PATH = os.path.join(OUTPUT_FOLDER, "velodyne/{0:06}.bin")
LABEL_PATH = os.path.join(OUTPUT_FOLDER, "labels/{0:06}.txt")
IMAGE_PATH = os.path.join(OUTPUT_FOLDER, "data/{0:06}.png")
CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, "calib/{0:06}.txt")


def transform_points(transform, points):
    """
    Given a 4x4 transformation matrix, transform an array of 3D points.
    Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]
    """
    # Needed foramt: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
    # the point matrix.

    points = points.transpose()
    # Add 0s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[0,..0]]
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # Point transformation
    points = np.mat(transform.get_matrix()) * points
    # Return all but last row
    return points[0:3].transpose()


def bbox_2d_from_agent(
    agent, intrinsic_mat, extrinsic_mat, ext, bbox_transform, agent_transform, rotRP
):  # rotRP expects point to be in Kitti lidar format
    """Creates bounding boxes for a given agent and camera/world calibration matrices.
    Returns the modified image that contains the screen rendering with drawn on vertices from the agent"""
    bbox = vertices_from_extension(ext)
    # transform the vertices respect to the bounding box transform
    bbox = transform_points(bbox_transform, bbox)
    # the bounding box transform is respect to the agents transform
    # so let's transform the points relative to it's transform
    bbox = transform_points(agent_transform, bbox)
    # agents's transform is relative to the world, so now,
    # bbox contains the 3D bounding box vertices relative to the world
    # Additionally, you can logging.info these vertices to check that is working
    # Store each vertex 2d points for drawing bounding boxes later
    vertices_pos2d = vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat)
    return vertices_pos2d


def point_in_canvas(pos):
    """Return true if point is in canvas"""
    if (
        (pos[0] >= 0)
        and (pos[0] < WINDOW_HEIGHT)
        and (pos[1] >= 0)
        and (pos[1] < WINDOW_WIDTH)
    ):
        return True
    return False


def draw_rect(array, pos, size, color):
    """Draws a rect"""
    point_0 = (pos[0] - size / 2, pos[1] - size / 2)
    point_1 = (pos[0] + size / 2, pos[1] + size / 2)
    if point_in_canvas(point_0) and point_in_canvas(point_1):
        for i in range(size):
            for j in range(size):
                array[int(point_0[0] + i), int(point_0[1] + j)] = color


def point_is_occluded(point, vertex_depth, depth_map):
    """Checks whether or not the four pixels directly around the given point has less depth than the given vertex depth
    If True, this means that the point is occluded.
    """
    y, x = map(int, point)
    from itertools import product

    neigbours = product((1, -1), repeat=2)
    is_occluded = []
    for dy, dx in neigbours:
        if point_in_canvas((dy + y, dx + x)):
            # If the depth map says the pixel is closer to the camera than the actual vertex
            if depth_map[y + dy, x + dx] < vertex_depth:
                is_occluded.append(True)
            else:
                is_occluded.append(False)
    # Only say point is occluded if all four neighbours are closer to camera than vertex
    return all(is_occluded)


def calculate_occlusion_stats(image, vertices_pos2d, depth_map, draw_vertices=True):
    """Draws each vertex in vertices_pos2d if it is in front of the camera
    The color is based on whether the object is occluded or not.
    Returns the number of visible vertices and the number of vertices outside the camera.
    """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    for y_2d, x_2d, vertex_depth in vertices_pos2d:
        # if the point is in front of the camera but not too far away
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas(
            (y_2d, x_2d)
        ):
            is_occluded = point_is_occluded((y_2d, x_2d), vertex_depth, depth_map)
            if is_occluded:
                vertex_color = OCCLUDED_VERTEX_COLOR
            else:
                num_visible_vertices += 1
                vertex_color = VISIBLE_VERTEX_COLOR
            if draw_vertices:
                draw_rect(image, (y_2d, x_2d), 4, vertex_color)
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
    array = array[:, :, :3]  # Take only RGB
    array = array[:, :, ::-1]  # BGR
    array = array.astype(np.float32)  # 2ms
    gray_depth = (
        array[:, :, 0] + array[:, :, 1] * 256.0 + array[:, :, 2] * 256.0 * 256.0
    ) / (
        (256.0 * 256.0 * 256.0) - 1
    )  # 2.5ms
    gray_depth = 1000 * gray_depth
    return gray_depth


def midpoint_from_agent_location(array, location, extrinsic_mat, intrinsic_mat):
    # Calculate the midpoint of the bottom chassis
    # This is used since kitti treats this point as the location of the car
    midpoint_vector = np.array(
        [[location.x], [location.y], [location.z], [1.0]]  # [[X,  # Y,  # Z,  # 1.0]]
    )
    transformed_3d_midpoint = proj_to_camera(midpoint_vector, extrinsic_mat)
    return transformed_3d_midpoint


def calc_projected_2d_bbox(vertices_pos2d):
    """Takes in all vertices in pixel projection and calculates min and max of all x and y coordinates.
    Returns left top, right bottom pixel coordinates for the 2d bounding box as a list of four values.
    Note that vertices_pos2d contains a list of (y_pos2d, x_pos2d) tuples, or None
    """

    legal_pos2d = list(filter(lambda x: x is not None, vertices_pos2d))
    y_coords, x_coords = [int(x[0][0]) for x in legal_pos2d], [
        int(x[1][0]) for x in legal_pos2d
    ]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return [min_x, min_y, max_x, max_y]


class KittiDescriptor:
    # This class is responsible for storing a single datapoint for the kitti 3d object detection task
    def __init__(
        self,
        type=None,
        bbox=None,
        dimensions=None,
        location=None,
        rotation_y=None,
        extent=None,
    ):
        self.type = type
        self.truncated = 0
        self.occluded = 0
        self.alpha = -10
        self.bbox = bbox
        self.dimensions = dimensions
        self.location = location
        self.rotation_y = rotation_y
        self.extent = extent
        self._valid_classes = [
            "Car",
            "Van",
            "Truck",
            "Pedestrian",
            "Person_sitting",
            "Cyclist",
            "Tram",
            "Misc",
            "DontCare",
        ]

    def set_type(self, obj_type: str):
        assert obj_type in self._valid_classes, "Object must be of types {}".format(
            self._valid_classes
        )
        self.type = obj_type

    def set_truncated(self, truncated: float):
        assert (
            0 <= truncated <= 1
        ), """Truncated must be Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries """
        self.truncated = truncated

    def set_occlusion(self, occlusion: int):
        assert occlusion in range(
            0, 4
        ), """Occlusion must be Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown"""
        self._occluded = occlusion

    def set_alpha(self, alpha: float):
        assert -pi <= alpha <= pi, "Alpha must be in range [-pi..pi]"
        self.alpha = alpha

    def set_bbox(self, bbox: List[int]):
        assert (
            len(bbox) == 4
        ), """ Bbox must be 2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates (two points)"""
        self.bbox = bbox

    def set_3d_object_dimensions(self, bbox_extent):
        # Bbox extent consists of x,y and z.
        # The bbox extent is by Carla set as
        # x: length of vehicle (driving direction)
        # y: to the right of the vehicle
        # z: up (direction of car roof)
        # However, Kitti expects height, width and length (z, y, x):
        height, width, length = bbox_extent.z, bbox_extent.x, bbox_extent.y
        # Since Carla gives us bbox extent, which is a half-box, multiply all by two
        self.extent = (height, width, length)
        self.dimensions = "{} {} {}".format(2 * height, 2 * width, 2 * length)

    def set_3d_object_location(self, obj_location):
        """TODO: Change this to
        Converts the 3D object location from CARLA coordinates and saves them as KITTI coordinates in the object
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ▲   ▲ x
        |  /
        | /
        |/____> y
        This is a left-handed coordinate system, with x being forward, y to the right and z up
        See also https://github.com/carla-simulator/carla/issues/498
        However, the camera coordinate system for KITTI is defined as
            ▲ z
           /
          /
         /____> x
        |
        |
        |
        ▼
        y
        This is a right-handed coordinate system with z being forward, x to the right and y down
        Therefore, we have to make the following changes from Carla to Kitti
        Carla: X   Y   Z
        KITTI:-X  -Y   Z
        """
        # Object location is four values (x, y, z, w). We only care about three of them (xyz)
        x, y, z = [float(x) for x in obj_location][0:3]
        assert None not in [
            self.extent,
            self.type,
        ], "Extent and type must be set before location!"
        if self.type == "Pedestrian":
            # Since the midpoint/location of the pedestrian is in the middle of the agent, while for car it is at the bottom
            # we need to subtract the bbox extent in the height direction when adding location of pedestrian.
            y -= self.extent[0]
        # Convert from Carla coordinate system to KITTI
        # This works for AVOD (image)
        # x *= -1
        # y *= -1
        self.location = " ".join(map(str, [y, -z, x]))
        # self.location = " ".join(map(str, [-x, -y, z]))
        # This works for SECOND (lidar)
        # self.location = " ".join(map(str, [z, x, y]))
        # self.location = " ".join(map(str, [z, x, -y]))

    def set_rotation_y(self, rotation_y: float):
        assert (
            -pi <= rotation_y <= pi
        ), "Rotation y must be in range [-pi..pi] - found {}".format(rotation_y)
        self.rotation_y = rotation_y

    def __str__(self):
        """Returns the kitti formatted string of the datapoint if it is valid (all critical variables filled out), else it returns an error."""
        if self.bbox is None:
            bbox_format = " "
        else:
            bbox_format = " ".join([str(x) for x in self.bbox])

        return "{} {} {} {} {} {} {} {}".format(
            self.type,
            self.truncated,
            self.occluded,
            self.alpha,
            bbox_format,
            self.dimensions,
            self.location,
            self.rotation_y,
        )


def get_line(x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # print("Calculating line from {},{} to {},{}".format(x1,y1,x2,y2))
    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


def create_kitti_datapoint(
    agent,
    intrinsic_mat,
    extrinsic_mat,
    image,
    depth_image,
    player,
    rotRP,
    draw_3D_bbox=True,
):
    """Calculates the bounding box of the given agent, and returns a KittiDescriptor which describes the object to
    be labeled"""
    obj_type, agent_transform, bbox_transform, ext, location = transforms_from_agent(
        agent
    )

    if obj_type is None:
        logging.warning("Could not get bounding box for agent. Object type is None")
        return image, None
    vertices_pos2d = bbox_2d_from_agent(
        agent, intrinsic_mat, extrinsic_mat, ext, bbox_transform, agent_transform, rotRP
    )
    depth_map = depth_to_array(depth_image)
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
        image, vertices_pos2d, depth_map, draw_vertices=draw_3D_bbox
    )

    midpoint = midpoint_from_agent_location(
        image, location, extrinsic_mat, intrinsic_mat
    )
    # At least N vertices has to be visible in order to draw bbox
    if (
        num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER
        and num_vertices_outside_camera < MIN_VISIBLE_VERTICES_FOR_RENDER
    ):
        bbox_2d = calc_projected_2d_bbox(vertices_pos2d)
        area = calc_bbox2d_area(bbox_2d)
        if area < MIN_BBOX_AREA_IN_PX:
            logging.info("Filtered out bbox with too low area {}".format(area))
            return image, None
        if draw_3D_bbox:
            draw_3d_bounding_box(image, vertices_pos2d)
        from math import pi

        # xiu gai
        rotation_y = get_relative_rotation_y(agent, player) % pi  # 取余数

        datapoint = KittiDescriptor()
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(ext)
        datapoint.set_type(obj_type)
        datapoint.set_3d_object_location(midpoint)
        datapoint.set_rotation_y(rotation_y)
        return image, datapoint
    else:
        return image, None


def draw_3d_bounding_box(array, vertices_pos2d):
    """Draws lines from each vertex to all connected vertices"""
    # Shows which verticies that are connected so that we can draw lines between them
    # The key of the dictionary is the index in the bbox array, and the corresponding value is a list of indices
    # referring to the same array.
    vertex_graph = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
    }
    # Note that this can be sped up by not drawing duplicate lines
    for vertex_idx in vertex_graph:
        neighbour_idxs = vertex_graph[vertex_idx]
        from_pos2d = vertices_pos2d[vertex_idx]
        for neighbour_idx in neighbour_idxs:
            to_pos2d = vertices_pos2d[neighbour_idx]
            if from_pos2d is None or to_pos2d is None:
                continue
            y1, x1 = from_pos2d[0], from_pos2d[1]
            y2, x2 = to_pos2d[0], to_pos2d[1]
            # Only stop drawing lines if both are outside
            if not point_in_canvas((y1, x1)) and not point_in_canvas((y2, x2)):
                continue
            for x, y in get_line(x1, y1, x2, y2):
                if point_in_canvas((y, x)):
                    array[int(y), int(x)] = (255, 0, 0)


def degrees_to_radians(degrees):
    return degrees * math.pi / 180


def get_relative_rotation_y(agent, player):
    """Returns the relative rotation of the agent to the camera in yaw
    The relative rotation is the difference between the camera rotation (on car) and the agent rotation"""
    # We only car about the rotation for the classes we do detection on
    if agent.get_transform():
        rot_agent = agent.get_transform().rotation.yaw
        rot_car = player.get_transform().rotation.yaw
        return degrees_to_radians(rot_agent - rot_car)


def proj_to_camera(pos_vector, extrinsic_mat):
    # transform the points to camera
    # print("Multiplied {} matrix with {} vector".format(extrinsic_mat.shape, pos_vector.shape))
    transformed_3d_pos = np.dot(np.linalg.inv(extrinsic_mat), pos_vector)
    return transformed_3d_pos


def proj_to_2d(camera_pos_vector, intrinsic_mat):
    # transform the points to 2D
    cords_x_y_z = camera_pos_vector[:3, :]
    cords_y_minus_z_x = np.concatenate(
        [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]]
    )
    pos2d = np.dot(intrinsic_mat, cords_y_minus_z_x)
    # normalize the 2D points
    pos2d = np.array([pos2d[0] / pos2d[2], pos2d[1] / pos2d[2], pos2d[2]])
    return pos2d


def vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat):
    """Accepts a bbox which is a list of 3d world coordinates and returns a list
    of the 2d pixel coordinates of each vertex.
    This is represented as a tuple (y, x, d) where y and x are the 2d pixel coordinates
    while d is the depth. The depth can be used for filtering visible vertices.
    """
    vertices_pos2d = []

    for vertex in bbox:
        pos_vector = vertex_to_world_vector(vertex)
        # Camera coordinates
        transformed_3d_pos = proj_to_camera(pos_vector, extrinsic_mat)
        # 2d pixel coordinates

        pos2d = proj_to_2d(transformed_3d_pos, intrinsic_mat)
        # The actual rendered depth (may be wall or other object instead of vertex)
        vertex_depth = pos2d[2]
        # x_2d, y_2d = WINDOW_WIDTH - pos2d[0],  WINDOW_HEIGHT - pos2d[1]
        x_2d, y_2d = pos2d[0], pos2d[1]
        vertices_pos2d.append((y_2d, x_2d, vertex_depth))

    return vertices_pos2d


def vertex_to_world_vector(vertex):
    """Returns the coordinates of the vector in correct carla world format (X,Y,Z,1)"""
    return np.array(
        [
            [vertex[0, 0]],  # [[X,
            [vertex[0, 1]],  # Y,
            [vertex[0, 2]],  # Z,
            [1.0],  # 1.0]]
        ]
    )


def vertices_from_extension(ext):
    """Extraxts the 8 bounding box vertices relative to (0,0,0)
    https://github.com/carla-simulator/carla/commits/master/Docs/img/vehicle_bounding_box.png
    8 bounding box vertices relative to (0,0,0)
    """
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


def transforms_from_agent(agent):
    """Returns the KITTI object type and transforms, locations and extension of the given agent"""
    if agent.type_id.find("walker") != -1:
        obj_type = "Pedestrian"
        agent_transform = agent.get_transform()
        bbox_transform = carla.Transform(
            agent.bounding_box.location, agent.bounding_box.rotation
        )
        ext = agent.bounding_box.extent
        location = agent.get_transform().location
    elif agent.type_id.find("vehicle") != -1:
        obj_type = "Car"
        agent_transform = agent.get_transform()
        bbox_transform = carla.Transform(
            agent.bounding_box.location, agent.bounding_box.rotation
        )
        ext = agent.bounding_box.extent
        location = agent.get_transform().location
    else:
        return (None, None, None, None, None)
    return obj_type, agent_transform, bbox_transform, ext, location


def calc_bbox2d_area(bbox_2d):
    """Calculate the area of the given 2d bbox
    Input is assumed to be xmin, ymin, xmax, ymax tuple
    """
    xmin, ymin, xmax, ymax = bbox_2d
    return (ymax - ymin) * (xmax - xmin)


def save_kitti_data(filename, datapoints):
    with open(filename, "w") as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)
    logging.info("Wrote kitti data to %s", filename)


def save_image_data(filename, image):
    logging.info("Wrote image data to %s", filename)
    image.save_to_disk(filename)


class SynchronyModel(object):
    def __init__(self):
        (
            self.world,
            self.init_setting,
            self.client,
            self.traffic_manager,
        ) = self._make_setting()
        self.blueprint_library = self.world.get_blueprint_library()
        self.non_player = []
        self.actor_list = []
        self.frame = None
        self.player = None
        self.captured_frame_no = self.current_captured_frame_num()
        self.sensors = []
        self._queues = []
        self.main_image = None
        self.depth_image = None
        self.point_cloud = None
        self.extrinsic = None
        self.intrinsic, self.my_camera = self._span_player()
        self._span_non_player()

    def __enter__(self):
        # set the sensor listener function
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def current_captured_frame_num(self):
        # Figures out which frame number we currently are on
        # This is run once, when we start the simulator in case we already have a dataset.
        # The user can then choose to overwrite or append to the dataset.
        label_path = os.path.join(OUTPUT_FOLDER, "label_2/")
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith(".txt")]
        )
        print(num_existing_data_files)
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(
                OUTPUT_FOLDER
            )
        )
        if answer.upper() == "O":
            logging.info("Resetting frame number to 0 and overwriting existing")
            # Overwrite the data
            return 0
        logging.info(
            "Continuing recording data on frame number {}".format(
                num_existing_data_files
            )
        )
        return num_existing_data_files

    def tick(self, timeout):
        # Drive the simulator to the next frame and get the data of the current frame
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

    def __exit__(self, *args, **kwargs):
        # cover the world settings
        self.world.apply_settings(self.init_setting)

    def _make_setting(self):
        client = carla.Client("localhost", 2000)
        client.set_timeout(5.0)
        client.load_world('Town10')
        client.reload_world()
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        traffic_manager.set_synchronous_mode(True)
        # synchrony model and fixed time step
        init_setting = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 steps  per second
        world.apply_settings(settings)
        return world, init_setting, client, traffic_manager

    def _span_player(self):
        """create our target vehicle"""
        my_vehicle_bp = random.choice(
            self.blueprint_library.filter("vehicle.lincoln.*")
        )

        success = False
        my_vehicle = None
        while not success:
            try:
                # location = carla.Location(70, 13, 0.5)
                # rotation = carla.Rotation(0, 180, 0)
                # transform_vehicle = carla.Transform(location, rotation)
                transform_vehicle = random.choice(
                    self.world.get_map().get_spawn_points()
                )
                my_vehicle = self.world.spawn_actor(my_vehicle_bp, transform_vehicle)
                success = True
            except:
                pass
        my_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        k, my_camera = self._span_sensor(my_vehicle)
        self.actor_list.append(my_vehicle)
        self.player = my_vehicle
        return k, my_camera

    def _span_sensor(self, player):
        """create camera, depth camera and lidar and attach to the target vehicle"""
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_d_bp = self.blueprint_library.find("sensor.camera.depth")
        lidar_bp = self.blueprint_library.find("sensor.lidar.ray_cast")

        camera_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
        camera_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))
        camera_bp.set_attribute("fov", "90")

        camera_d_bp.set_attribute("image_size_x", str(WINDOW_WIDTH))
        camera_d_bp.set_attribute("image_size_y", str(WINDOW_HEIGHT))
        camera_d_bp.set_attribute("fov", "90")

        lidar_bp.set_attribute("range", "50")
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar_bp.set_attribute("upper_fov", "2")
        lidar_bp.set_attribute("lower_fov", "-26.8")
        lidar_bp.set_attribute("points_per_second", "320000")
        lidar_bp.set_attribute("channels", "32")

        transform_sensor = carla.Transform(
            carla.Location(x=0, y=0, z=CAMERA_HEIGHT_POS),
            carla.Rotation(0, 0, 0),
        )

        my_camera = self.world.spawn_actor(
            camera_bp, transform_sensor, attach_to=player
        )
        my_camera_d = self.world.spawn_actor(
            camera_d_bp, transform_sensor, attach_to=player
        )
        my_lidar = self.world.spawn_actor(lidar_bp, transform_sensor, attach_to=player)
        self.world.tick()
        self.actor_list.append(my_camera)
        self.actor_list.append(my_camera_d)
        self.actor_list.append(my_lidar)
        self.sensors.append(my_camera)
        self.sensors.append(my_camera_d)
        self.sensors.append(my_lidar)

        # camera intrinsic
        k = np.identity(3)
        k[0, 2] = WINDOW_WIDTH_HALF
        k[1, 2] = WINDOW_HEIGHT_HALF
        f = WINDOW_WIDTH / (2.0 * math.tan(90.0 * math.pi / 360.0))
        k[0, 0] = k[1, 1] = f
        return k, my_camera

    def _span_non_player(self):
        """create autonomous vehicles and people"""
        blueprints = self.world.get_blueprint_library().filter(FILTERV)
        blueprints = [
            x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
        ]
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        if NUM_OF_VEHICLES < number_of_spawn_points:
            random.shuffle(spawn_points)
            number_of_vehicles = NUM_OF_VEHICLES
        elif NUM_OF_VEHICLES > number_of_spawn_points:
            msg = "requested %d vehicles, but could only find %d spawn points"
            logging.warning(msg, NUM_OF_VEHICLES, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(
                    blueprint.get_attribute("color").recommended_values
                )
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    blueprint.get_attribute("driver_id").recommended_values
                )
                blueprint.set_attribute("driver_id", driver_id)
            blueprint.set_attribute("role_name", "autopilot")

            # spawn the cars and set their autopilot
            batch.append(SpawnActor(blueprint, transform))

        vehicles_id = []
        for response in self.client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_id.append(response.actor_id)
        vehicle_actors = self.world.get_actors(vehicles_id)
        self.non_player.extend(vehicle_actors)
        self.actor_list.extend(vehicle_actors)

        for i in vehicle_actors:
            i.set_autopilot(True, self.traffic_manager.get_port())

        blueprintsWalkers = self.world.get_blueprint_library().filter(FILTERW)
        percentagePedestriansCrossing = (
            0.0  # how many pedestrians will walk through the road
        )
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(NUM_OF_WALKERS):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc != None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        percentagePedestriansRunning = 0.3
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            # set the max speed
            if walker_bp.has_attribute("speed"):
                if random.random() > percentagePedestriansRunning:
                    # walking
                    walker_speed.append(
                        walker_bp.get_attribute("speed").recommended_values[1]
                    )
                else:
                    # running
                    walker_speed.append(
                        walker_bp.get_attribute("speed").recommended_values[2]
                    )
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        walkers_list = []
        all_id = []
        walkers_id = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find(
            "controller.ai.walker"
        )
        for i in range(len(walkers_list)):
            batch.append(
                SpawnActor(
                    walker_controller_bp, carla.Transform(), walkers_list[i]["id"]
                )
            )
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_id)

        for i in range(len(walkers_list)):
            walkers_id.append(walkers_list[i]["id"])
        walker_actors = self.world.get_actors(walkers_id)
        self.non_player.extend(walker_actors)
        self.actor_list.extend(all_actors)
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(
                self.world.get_random_location_from_navigation()
            )
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print(
            "spawned %d walkers and %d vehicles, press Ctrl+C to exit."
            % (len(walkers_id), len(vehicles_id))
        )

    def _save_training_files(self, datapoints, point_cloud):
        """Save data in Kitti dataset format"""
        logging.info(
            "Attempting to save at frame no {}, frame no: {}".format(
                self.frame, self.captured_frame_no
            )
        )
        groundplane_fname = GROUNDPLANE_PATH.format(self.captured_frame_no)
        lidar_fname = LIDAR_PATH.format(self.captured_frame_no)
        kitti_fname = LABEL_PATH.format(self.captured_frame_no)
        img_fname = IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = CALIBRATION_PATH.format(self.captured_frame_no)

        # save_groundplanes(groundplane_fname, self.player, LIDAR_HEIGHT_POS)
        # save_ref_files(OUTPUT_FOLDER, self.captured_frame_no)
        save_image_data(img_fname, self.main_image)
        save_kitti_data(kitti_fname, datapoints)

        # save_calibration_matrices(calib_filename, self.intrinsic, self.extrinsic)

        # save_lidar_data(lidar_fname, point_cloud)

    def generate_datapoints(self, image):
        """Returns a list of datapoints (labels and such) that are generated this frame together with the main image
        image"""
        datapoints = []
        image = image.copy()
        # Remove this
        rotRP = np.identity(3)
        # Stores all datapoints for the current frames
        for agent in self.non_player:
            if GEN_DATA:
                image, kitti_datapoint = create_kitti_datapoint(
                    agent,
                    self.intrinsic,
                    self.extrinsic,
                    image,
                    self.depth_image,
                    self.player,
                    rotRP,
                )
                if kitti_datapoint:
                    datapoints.append(kitti_datapoint)

        return image, datapoints


def main():

    with SynchronyModel() as sync_mode:
        try:
            step = 1
            while True:
                (
                    snapshot,
                    sync_mode.main_image,
                    sync_mode.depth_image,
                    sync_mode.point_cloud,
                ) = sync_mode.tick(timeout=2.0)

                image = image_converter.to_rgb_array(sync_mode.main_image)
                sync_mode.extrinsic = np.mat(
                    sync_mode.my_camera.get_transform().get_matrix()
                )
                image, datapoints = sync_mode.generate_datapoints(image)

                if datapoints and step % 100 is 0:
                    data = np.copy(
                        np.frombuffer(
                            sync_mode.point_cloud.raw_data, dtype=np.dtype("f4")
                        )
                    )
                    data = np.reshape(data, (int(data.shape[0] / 4), 4))
                    # Isolate the 3D data
                    points = data[:, :-1]
                    # transform to car space
                    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
                    # points = np.dot(sync_mode.player.get_transform().get_matrix(), points.T).T
                    # points = points[:, :-1]
                    # points[:, 2] -= LIDAR_HEIGHT_POS
                    sync_mode._save_training_files(datapoints, points)
                    sync_mode.captured_frame_no += 1

                step = step + 1
        finally:
            print("destroying actors.")
            for actor in sync_mode.actor_list:
                actor.destroy()
            print("done.")


if __name__ == "__main__":
    main()
