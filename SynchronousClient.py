import random
import numpy as np

import carla
from clientlib.camera_utils import CustomCamera
from clientlib.lidar_utils import CustomLidar


class SynchronousClient:
    def __init__(self, cfg, args=None):
        self.cfg = cfg
        if args:
            self.client = carla.Client(str(args.host), int(args.port))
        else:
            self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.manager = self.client.get_trafficmanager(8000)

        self.frames_per_second = 20
        self.set_synchronous_mode(True)

        self.number_of_cars = int(self.cfg["CARLA_CONFIG"]["NUM_OF_VEHICLES"])
        self.number_of_walkers = int(self.cfg["CARLA_CONFIG"]["NUM_OF_WALKERS"])

        self.ego = None
        self.spectator = None

        self.image_x = self.cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"]
        self.image_y = self.cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"]

        self.fov = 90
        self.sensor_tick = 0.1
        self.tick = -1

    def set_synchronous_mode(self, synchronous_mode):

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 1 / self.frames_per_second
        self.world.apply_settings(settings)
        self.manager.set_synchronous_mode(synchronous_mode)

    def setup_cars(self):

        car_bps = self.world.get_blueprint_library().filter("vehicle.*")
        car_bp_sample = []
        for _ in range(self.number_of_cars):
            car_bp_sample.append(random.choice(car_bps))
        location = random.sample(
            self.world.get_map().get_spawn_points(), self.number_of_cars
        )
        success = False
        while not success:
            try:
                self.ego = self.world.spawn_actor(car_bps[0], location[0])
                self.ego.set_autopilot(True, self.manager.get_port())
                success = True
            except:
                pass

        for i in range(1, self.number_of_cars):
            current_car = self.world.spawn_actor(car_bp_sample[i], location[i])
            current_car.set_autopilot(True, self.manager.get_port())

    def update_spectator(self, transform=[-5.5, 0.0, 2.8, -15.0, 0.0, 0.0]):
        """transform: = [x, y, z, pitch, yaw, roll]"""

        specatator_vehicle_transform = carla.Transform(
            location=carla.Location(*transform[0:3]),
            rotation=carla.Rotation(*transform[3:6]),
        )
        specatator_vehicle_matrix = specatator_vehicle_transform.get_matrix()
        vehicle_world_matrix = self.ego.get_transform().get_matrix()
        specatator_world_matrix = np.dot(
            vehicle_world_matrix, specatator_vehicle_matrix
        )
        specatator_world = np.dot(
            specatator_world_matrix,
            np.transpose(np.array([[*transform[0:3], 1.0]], dtype=np.dtype("float32"))),
        )
        specatator_rotation = self.ego.get_transform().rotation
        specatator_rotation.pitch = -15
        spectator_transform = carla.Transform(
            location=carla.Location(*specatator_world[0:3, 0]),
            rotation=specatator_rotation,
        )
        self.spectator.set_transform(spectator_transform)

    def setup_spectator(self):

        self.spectator = self.world.get_spectator()
        self.update_spectator()

    def setup_camera(self, transform, log_dir="training/data", suffix=""):
        """transform: = [x, y, z, pitch, yaw, roll]"""

        camera_options = {
            "image_size_x": self.image_x,
            "image_size_y": self.image_y,
            "fov": self.fov,
            "sensor_tick": self.sensor_tick,
            "motion_blur_intensity": 0.0,
        }

        camera_location = carla.Location(*transform[0:3])
        camera_rotation = carla.Rotation(*transform[3:6])
        camera_transform = carla.Transform(
            location=camera_location, rotation=camera_rotation
        )
        return CustomCamera(
            self.world,
            camera_transform,
            self.ego,
            log_dir,
            suffix=suffix,
            with_bbox=True,
            **camera_options
        )

    def setup_lidar(self, transform, log_dir="trainind/velodyne", **options):
        """transform: = [x, y, z, pitch, yaw, roll]"""

        lidar_location = carla.Location(*transform[0:3])
        lidar_rotation = carla.Rotation(*transform[3:6])
        lidar_transform = carla.Transform(
            location=lidar_location, rotation=lidar_rotation
        )
        return CustomLidar(self.world, lidar_transform, self.ego, log_dir, **options)
