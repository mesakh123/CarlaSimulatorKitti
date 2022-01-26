from DataSave import DataSave
from carlautils.SynchronyModel import SynchronyModel
from config import cfg_from_yaml_file
from carlautils.data_utils import objects_filter
import argparse


def main(args):
    cfg = cfg_from_yaml_file("configs.yaml")
    model = SynchronyModel(cfg)
    dtsave = DataSave(cfg, args.kitti_only)
    try:
        model.set_synchrony()
        model.spawn_actors()
        model.set_actors_route()
        model.spawn_agent()
        model.sensor_listen()
        step = 0
        STEP = cfg["SAVE_CONFIG"]["STEP"]
        while True:
            if step % STEP == 0:
                data = model.tick()
                data = objects_filter(data)
                dtsave.save_training_files(data)
                print(step / STEP)
            else:
                model.world.tick()
            step += 1
    finally:
        model.setting_recover()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="CARLA Automatic Control Client")
    argparser.add_argument(
        "--kitti-only",
        action="store_true",
        dest="kitti_only",
        help="Only save KIITTI dataset format (images and KITTI labels)",
    )
    args = argparser.parse_args()
    main(args)
