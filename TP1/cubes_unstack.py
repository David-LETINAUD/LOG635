import cozmo
from cozmo.util import degrees
import time

def cube_unstack(robot: cozmo.robot.Robot):
    lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    cubes = robot.world.wait_until_observe_num_objects(num=2, object_type=cozmo.objects.LightCube, timeout=60)
    lookaround.stop()

    if len(cubes) < 2:
        print("Error: need 2 Cubes but only found", len(cubes), "Cube(s)")
    else:
        robot.pickup_object(cubes[1], num_retries=3).wait_for_completed()
        robot.turn_in_place(degrees(90)).wait_for_completed()
        robot.place_object_on_ground_here(cubes[1])


#cozmo.run_program(cozmo_program, use_viewer=True)
