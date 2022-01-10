from DataSave import DataSave
from SynchronyModel import SynchronyModel
from config import cfg_from_yaml_file
from data_utils import objects_filter

import logging
import pygame
import os
import sys
import argparse
print("Import ",os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/CarlaSimulatorKitti/carla')
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/CarlaSimulatorKitti/carla')
except IndexError:
    raise RuntimeError(
        'cannot import carla, make sure numpy package is installed')
import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

from utils.hud import HUD
from utils.keyboard import KeyboardControl
from utils.world import World
import random


import threading
from time import sleep


save = False


def save_data(model,dtsave):
    while True:
        if save:        
            data = model.tick()
            data = objects_filter(data)
            dtsave.save_training_files(data)
        sleep(1)
            
    




def main(args):
    cfg = cfg_from_yaml_file("configs.yaml")
    model = SynchronyModel(cfg)
    dtsave = DataSave(cfg)
    
    
    pygame.init()
    pygame.font.init()
    world = None
    args.sync = True
    args.behaviour = "Basic"

    try:
        model.set_synchrony()
        model.spawn_actors()
        model.set_actors_route()
        model.spawn_agent()
        model.sensor_listen()
        step = 0
        STEP = cfg["SAVE_CONFIG"]["STEP"]
        image_width = cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_x"]
        image_height = cfg["SENSOR_CONFIG"]["RGB"]["ATTRIBUTE"]["image_size_y"]
        
        args.width, args.height = image_width, image_height
        
        display = pygame.display.set_mode(
            (image_width,image_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        
        hud = HUD(image_width, image_height)
        world = World(model.world,hud,args)
        
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
        thread = threading.Thread(target=save_data,args=(model,dtsave,))
        thread.start()
        
        while True:
            clock.tick()
            model.world.tick()
            world.world.tick()
            if controller.parse_events():
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            if agent.done():
                if args.loop:
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification("The target has been reached, searching for another target", seconds=4.0)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break

            if step % STEP ==0:
                
                print(step / STEP)
            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)
            step+=1
    finally:
        model.setting_recover()
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            world.destroy()
        pygame.quit()


if __name__ == '__main__':
    
    
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()


    log_level = logging.DEBUG if args.debug else logging.INFO
    #logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    #logging.info('listening to server %s:%s', args.host, args.port)


    main(args)
