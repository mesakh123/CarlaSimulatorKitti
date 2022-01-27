# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================


import carla
from carla import ColorConverter as cc
import sys
from .collision_sensor import CollisionSensor
from .lane_sensor import LaneInvasionSensor
from .gns_sensor import GnssSensor
from .camera_manager import CameraManager
from .global_functions import *
import random
from .global_functions import *
from utils.custom_classes import *


classes = {
    "kitti": KITTI_CLASSES,
    "custom": CUSTOM_CLASSES,
    "coco": COCO_CLASSES,
    "carla": CARLA_CLASSES,
}


class World(object):
    """Class representing the surrounding environment"""

    def __init__(self, carla_world, hud, args, remote_player: None):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print("RuntimeError: {}".format(error))
            print("  The server could not send the OpenDRIVE (.xodr) file:")
            print(
                "  Make sure it exists, has the same name of your town, and is correct."
            )
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self.classes = classes[args.classes] if \
            args.classes in classes else classes["kitti"]

        self.restart(args, remote_player)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args, remote_player: None):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = (
            self.camera_manager.transform_index
            if self.camera_manager is not None
            else 0
        )

        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter)
        )
        blueprint.set_attribute("role_name", "hero")
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print("There are no spawn points available in your map/town.")
                print("Please add some Vehicle Spawn Point to your UE4 scene.")
                sys.exit(1)
            if remote_player is not None:
                self.player = remote_player
            else:
                spawn_points = self.map.get_spawn_points()
                spawn_point = (
                    random.choice(spawn_points) if spawn_points else carla.Transform()
                )
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, args, self.classes)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification("Weather: %s" % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player,
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()
