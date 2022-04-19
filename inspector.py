from DataSave import DataSave
from carlautils.SynchronyModel import SynchronyModel
from config import cfg_from_yaml_file
from carlautils.data_utils import objects_filter



import logging
import pygame
import os
import sys
import argparse

print(
    "Import ",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    + "/CarlaSimulatorKitti/carla",
)
try:
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/CarlaSimulatorKitti/carla"
    )
except IndexError:
    raise RuntimeError("cannot import carla, make sure numpy package is installed")

from carla import ColorConverter as cc

from agents.navigation.behavior_agent import (
    BehaviorAgent,
)  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

from utils.hud import HUD
from utils.keyboard import KeyboardControl
from utils.world import World
import random


import threading
import time

save = False
model = None
dtsave = None
step = 0
STEP = 0

def save_data():
    global save, model, dtsave, step, STEP
    while True:
        if step % STEP == 0:
            time.sleep(1)
            with threading.Lock():
                data = model.tick()
                data = objects_filter(model.player, data)
                dtsave.save_training_files(data)
                save = False
                print("Step {} saved".format(int(step/STEP)))


def main(args):
    global save, model, dtsave, tasks, step, STEP
    cfg = cfg_from_yaml_file("configs.yaml")
    model = SynchronyModel(cfg, args)
    dtsave = DataSave(cfg, args.kitti_only)

    pygame.init()
    pygame.font.init()
    world = None
    args.sync = True
    args.behaviour = "Basic"

    step = 0
    STEP = cfg["SAVE_CONFIG"]["STEP"]
    image_width = cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"]
    image_height = cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"]

    try:
        model.set_synchrony()
        model.spawn_actors()
        model.set_actors_route()
        model.spawn_agent()
        model.sensor_listen()

        args.width, args.height = image_width, image_height

        display = pygame.display.set_mode(
            (image_width, image_height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        hud = HUD(image_width, image_height)

        world = World(model.world, hud, args, model.player)

        controller = KeyboardControl(world)
        if args.agent == "Basic":
            agent = BasicAgent(world.player)
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)
        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)

        clock = pygame.time.Clock()
        th = threading.Thread(target=save_data, args=())
        th.setDaemon(True)
        th.start()

        while True:
            clock.tick()
            # model.world.tick()
            world.world.tick()
            if controller.parse_events():
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            if agent.done():
                if args.loop:
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification(
                        "The target has been reached, searching for another target",
                        seconds=4.0,
                    )
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break

            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)
            step += 1
    finally:
        model.setting_recover()
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            try:
                world.destroy()
            except:
                pass

        th.join()
        pygame.quit()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="CARLA Automatic Control Client")
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="Print debug information",
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )

    argparser.add_argument(
        "--model-host",
        default=None,
        help="IP of the host server (default: None)",
    )
    argparser.add_argument(
        "--model-port",
        default=None,
        type=int,
        help="TCP port to listen to model_host (default: None)",
    )

    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="1280x720",
        help="Window resolution (default: 1280x720)",
    )
    argparser.add_argument(
        "--sync", action="store_true", help="Synchronous mode execution"
    )
    argparser.add_argument(
        "--filter",
        metavar="PATTERN",
        default="vehicle.*",
        help='Actor filter (default: "vehicle.*")',
    )
    argparser.add_argument(
        "-l",
        "--loop",
        action="store_true",
        dest="loop",
        help="Sets a new random destination upon reaching the previous one (default: False)",
    )
    argparser.add_argument(
        "--predict",
        action="store_true",
        dest="predict",
        help="Predict images on pygame (default: False)",
    )
    argparser.add_argument(
        "--kitti-only",
        action="store_true",
        dest="kitti_only",
        help="Only save KIITTI dataset format (images and KITTI labels)",
    )
    argparser.add_argument(
        "-a",
        "--agent",
        type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Behavior",
    )
    argparser.add_argument(
        "-b",
        "--behavior",
        type=str,
        choices=["cautious", "normal", "aggressive"],
        help="Choose one of the possible agent behaviors (default: normal) ",
        default="normal",
    )
    argparser.add_argument(
        "-s",
        "--seed",
        help="Set seed for repeating executions (default: None)",
        default=None,
        type=int,
    )
    argparser.add_argument(
        "--conf",
        help="Set confidence of bbox (default: 0.6)",
        default=0.6,
        dest="conf",
        type=float,
    )
    argparser.add_argument(
        "-c",
        "--classes",
        type=str,
        choices=["carla", "kitti", "coco", "custom"],
        help="Choose one of the classes list (default: kitti) ",
        default="carla",
    )
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    # logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # logging.info('listening to server %s:%s', args.host, args.port)

    main(args)
