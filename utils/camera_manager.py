# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc
import weakref
import numpy as np
import pygame
import random
import cv2


class CameraManager(object):
    """Class for camera management"""

    def __init__(self, parent_actor, hud, predictions=None):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.predictions = predictions
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (
                carla.Transform(
                    carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)
                ),
                attachment.SpringArm,
            ),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (
                carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),
                attachment.SpringArm,
            ),
            (
                carla.Transform(
                    carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)
                ),
                attachment.SpringArm,
            ),
            (
                carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)),
                attachment.Rigid,
            ),
        ]
        self.transform_index = 1
        self.sensors = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB"],
            ["sensor.camera.depth", cc.Raw, "Camera Depth (Raw)"],
            ["sensor.camera.depth", cc.Depth, "Camera Depth (Gray Scale)"],
            [
                "sensor.camera.depth",
                cc.LogarithmicDepth,
                "Camera Depth (Logarithmic Gray Scale)",
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.Raw,
                "Camera Semantic Segmentation (Raw)",
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.CityScapesPalette,
                "Camera Semantic Segmentation (CityScapes Palette)",
            ],
            ["sensor.lidar.ray_cast", None, "Lidar (Ray-Cast)"],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                blp.set_attribute("image_size_x", str(hud.dim[0]))
                blp.set_attribute("image_size_y", str(hud.dim[1]))
            elif item[0].startswith("sensor.lidar"):
                blp.set_attribute("range", "50")
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = (
            True
            if self.index is None
            else (
                force_respawn or (self.sensors[index][0] != self.sensors[self.index][0])
            )
        )
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1],
            )

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image)
            )
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def __init_calibration_matrix(self):
        image_x = float(self.sensor.attributes["image_size_x"])
        image_y = float(self.sensor.attributes["image_size_y"])
        fov = float(self.sensor.attributes["fov"])
        calibration = np.identity(3)
        calibration[0, 2] = image_x / 2.0
        calibration[1, 2] = image_y / 2.0
        calibration[0, 0] = calibration[1, 1] = image_x / (
            2.0 * np.tan(fov * np.pi / 360.0)
        )
        self.image_width = image_x
        self.image_height = image_y
        self.fov = fov
        return calibration

    def get_calibration_matrix(self):
        return self.calibration

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(
                lidar_data
            )  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            cv2.imwrite("{}.png".format(random.randint(1, 14551)), array)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)
