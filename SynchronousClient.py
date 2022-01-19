import random
import numpy as np
import queue

import carla
from clientlib.camera_utils import CustomCamera
from clientlib.lidar_utils import CustomLidar

from data_utils import camera_intrinsic, filter_by_distance


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

        self.actors = {"non_agents": [], "walkers": [], "agents": [], "sensors": {}}
        self.data = {"sensor_data": {}, "environment_data": None}  # 记录每一帧的数据

    def set_synchronous_mode(self, synchronous_mode):

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 1 / self.frames_per_second
        self.world.apply_settings(settings)
        self.manager.set_synchronous_mode(synchronous_mode)

    def setup_cars_ori(self):

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

    def setting_recover(self):
        for agent in self.actors["agents"]:
            for sensor in self.actors["sensors"][agent]:
                sensor.destroy()
            agent.destroy()
        batch = []
        for actor_id in self.actors["non_agents"]:
            batch.append(carla.command.DestroyActor(actor_id))
        for walker_id in self.actors["walkers"]:
            batch.append(carla.command.DestroyActor(walker_id))
        self.client.apply_batch_sync(batch)
        self.world.apply_settings(self.init_settings)

    def setup_cars(self):
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        spawn_points = self.world.get_map().get_spawn_points()

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= self.number_of_cars:
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
            batch.append(carla.command.SpawnActor(blueprint, transform))

            for response in self.client.apply_batch_sync(batch):
                if response.error:
                    continue
                else:
                    self.actors["non_agents"].append(response.actor_id)

        # 生成行人actors
        blueprintsWalkers = self.world.get_blueprint_library().filter(
            "walker.pedestrian.*"
        )
        spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                continue
            else:
                self.actors["walkers"].append(response.actor_id)
        print(
            "spawn {} vehicles and {} walkers".format(
                len(self.actors["non_agents"]), len(self.actors["walkers"])
            )
        )
        self.world.tick()
        self.set_actors_route()
        self.set_ego()
        self.sensor_listen()
        self.setup_spectator()

    def set_actors_route(self):
        self.manager.set_global_distance_to_leading_vehicle(1.0)
        self.manager.set_synchronous_mode(True)
        vehicle_actors = self.world.get_actors(self.actors["non_agents"])
        for vehicle in vehicle_actors:
            vehicle.set_autopilot(True, self.manager.get_port())

        walker_controller_bp = self.world.get_blueprint_library().find(
            "controller.ai.walker"
        )
        batch = []
        for i in range(len(self.actors["walkers"])):
            batch.append(
                carla.command.SpawnActor(
                    walker_controller_bp, carla.Transform(), self.actors["walkers"][i]
                )
            )
        controllers_id = []
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                pass
            else:
                controllers_id.append(response.actor_id)
        self.world.set_pedestrians_cross_factor(0.2)

        for con_id in controllers_id:
            # start walker
            self.world.get_actor(con_id).start()
            # set walk to random point
            destination = self.world.get_random_location_from_navigation()
            self.world.get_actor(con_id).go_to_location(destination)
            # max speed
            self.world.get_actor(con_id).set_max_speed(10)

    def set_ego(self):
        car_bps = self.world.get_blueprint_library().filter("vehicle.*")
        spawn_points = self.world.get_map().get_spawn_points()

        car = random.choice(car_bps)

        success = False
        while not success:
            try:
                location = random.choice(spawn_points)
                self.ego = self.world.spawn_actor(car, location)
                success = True
            except:
                pass
        self.ego.set_autopilot(True, self.manager.get_port())

        self.actors["agents"].append(self.ego)

        self.actors["sensors"][self.ego] = []

        for sensor, config in self.cfg["SENSOR_CONFIG"].items():
            sensor_bp = self.world.get_blueprint_library().find(config["BLUEPRINT"])
            for attr, val in config["ATTRIBUTE"].items():
                sensor_bp.set_attribute(attr, str(val))
            trans_cfg = config["TRANSFORM"]
            transform = carla.Transform(
                carla.Location(
                    trans_cfg["location"][0],
                    trans_cfg["location"][1],
                    trans_cfg["location"][2],
                ),
                carla.Rotation(
                    trans_cfg["rotation"][0],
                    trans_cfg["rotation"][1],
                    trans_cfg["rotation"][2],
                ),
            )
            sensor = self.world.spawn_actor(sensor_bp, transform, attach_to=self.ego)
            self.actors["sensors"][self.ego].append(sensor)
        self.world.tick()

    def sensor_listen(self):
        for agent, sensors in self.actors["sensors"].items():
            self.data["sensor_data"][agent] = []
            for sensor in sensors:
                q = queue.Queue()
                self.data["sensor_data"][agent].append(q)
                sensor.listen(q.put)

    def tick(self):
        ret = {"environment_objects": None, "actors": None, "agents_data": {}}
        self.frame = self.world.tick()

        ret["environment_objects"] = self.world.get_environment_objects(
            carla.CityObjectLabel.Any
        )
        ret["actors"] = self.world.get_actors()
        image_width = self.cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"]
        image_height = self.cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"]
        for agent, dataQue in self.data["sensor_data"].items():
            data = [self._retrieve_data(q) for q in dataQue]
            assert all(x.frame == self.frame for x in data)
            ret["agents_data"][agent] = {}
            ret["agents_data"][agent]["sensor_data"] = data
            ret["agents_data"][agent]["intrinsic"] = camera_intrinsic(
                image_width, image_height
            )
            ret["agents_data"][agent]["extrinsic"] = np.mat(
                self.actors["sensors"][agent][0].get_transform().get_matrix()
            )
        filter_by_distance(
            ret, self.cfg["FILTER_CONFIG"]["PRELIMINARY_FILTER_DISTANCE"]
        )
        return ret

    def _retrieve_data(self, q):
        while True:
            data = q.get()
            if data.frame == self.frame:
                return data
