from DataSave import DataSave
from carlautils.SynchronyModel import SynchronyModel
from config import cfg_from_yaml_file
from carlautils.data_utils import objects_filter
import argparse
import time

def main(args):
    cfg = cfg_from_yaml_file("configs.yaml")
    model = SynchronyModel(cfg,args)
    dtsave = DataSave(cfg, args.kitti_only)
    STEP = cfg["SAVE_CONFIG"]["STEP"] if not args.frame_rate else int(args.frame_rate)

    time_limit = int(args.time_limit)*60 \
        if args and args.time_limit else None

    try:
        model.set_synchrony()
        model.spawn_actors()
        model.set_actors_route()
        model.spawn_agent()
        model.sensor_listen()
        step = 0
        start = time.time()

        print("Time limit ",time_limit)
        while True:
            #Set time limit
            if time_limit and \
                (int(time.time() - start) == \
                    time_limit):
                break

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
        "--res",
        metavar="WIDTHxHEIGHT",
        default=None,
        help="Window resolution i.e 1280x720 (default: 1224x370)",
    )
    argparser.add_argument(
        "-l",
        "--loop",
        action="store_true",
        dest="loop",
        help="Sets a new random destination upon reaching the previous one (default: False)",
    )
    argparser.add_argument(
        "--kitti-only",
        action="store_true",
        dest="kitti_only",
        help="Only save KIITTI dataset format (images and KITTI labels)",
    )
    argparser.add_argument(
        "--vehicle-num",
        default=50,
        type=int,
        help="Number of vehicle in the world",
    )    
    argparser.add_argument(
        "--time-limit",
        default=0,
        type=int,
        help="Set collect runtime timeout in minute (default : 0 (no timeout))",
    )
    argparser.add_argument(
        "--walker-num",
        default=20,
        type=int,
        help="Number of vehicle in the world",
    )
    argparser.add_argument(
        "--town",
        default="Town01",
        type=str,
        help="Selecting town, (default : Town01)",
        choices=['Town01','Town01_Opt', \
        'Town02','Town02_Opt','Town03','Town03_Opt','Town04','Town04_Opt',\
        'Town05','Town05_Opt','Town10HD','Town10HD_Opt']
    )
    argparser.add_argument(
        "--image-type",
        default="jpg",
        type=str,
        help="Image saved format (default : jpg)",
        choices=['jpg', 'png', 'jpeg']
    )
    argparser.add_argument(
        "--frame-rate",
        default=30,
        type=int,
        help="Save image on every 'frame-rate' frame (default : 30)",
    )

    #Coming class filtering

    
    args = argparser.parse_args()
    main(args)


