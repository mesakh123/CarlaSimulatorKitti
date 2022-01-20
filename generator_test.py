import sys
import os
import glob
import argparse
import logging

from SynchronousClient import SynchronousClient
from config import cfg_from_yaml_file
from clientlib.kitti_utils import generate_kitti_label_file

import time


def main(args):
    cfg = cfg_from_yaml_file("configs.yaml")

    world = None
    model = None
    try:
        model = SynchronousClient(cfg)
        tick = 0

        model.setup_cars()

        label_count = 1

        front = model.setup_camera([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        while True:
            tick += 1
            model.world.tick()
            model.update_spectator()
            front.save_data()
            if tick % (model.sensor_tick // (1 / model.frames_per_second)) == 0:
                generate_kitti_label_file(
                    ("%06d.txt" % label_count), model.world, front
                )
                label_count += 1
                time.sleep(1)
    finally:
        if world:
            vehicles = world.get_actors().filter("vehicle.*")
            for vehicle in vehicles:
                vehicle.destroy()
            model.set_synchronous_mode(False)


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

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    # logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    # logging.info('listening to server %s:%s', args.host, args.port)

    main(args)
