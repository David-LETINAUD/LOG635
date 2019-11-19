import cozmo
from cozmo.objects import LightCube1Id, LightCube2Id, LightCube3Id
import time

def cozmo_lights(robot: cozmo.robot.Robot):
    robot.say_text("Light up cubes").wait_for_completed()

    # get cube 1
    cube1 = robot.world.get_light_cube(LightCube1Id)
    # get cube 2
    cube2 = robot.world.get_light_cube(LightCube2Id)
    # get cube 3
    cube3 = robot.world.get_light_cube(LightCube3Id)

    if cube1 is not None:
        cube1.set_lights(cozmo.lights.red_light)
    else:
        cozmo.logger.warning("LightCube1Id cube is not connected - check the battery.")
    if cube2 is not None:
        cube2.set_lights(cozmo.lights.green_light)
    else:
        cozmo.logger.warning("LightCube2Id cube is not connected - check the battery.")
    if cube3 is not None:
        cube3.set_lights(cozmo.lights.blue_light)
    else:
        cozmo.logger.warning("LightCube3Id cube is not connected - check the battery.")
    return cube1, cube2, cube3

def cozmo_program(robot: cozmo.robot.Robot):
    c1, c2, c3 = cozmo_lights(robot)
    print("Cube 1: ", c1)
    print("------------")
    print("Cube 2: ", c2)
    print("------------")
    print("Cube 3: ", c3)
    
    while True:
        time.sleep(0.1)

cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True, force_viewer_on_top=True)
