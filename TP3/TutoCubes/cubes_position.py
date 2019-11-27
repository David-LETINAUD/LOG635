import cozmo
from cozmo.util import distance_mm

def cozmo_program(robot: cozmo.robot.Robot):
    lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    cubes = robot.world.wait_until_observe_num_objects(num=3, object_type=cozmo.objects.LightCube, timeout=60)
    lookaround.stop()

    if len(cubes) < 2:
        print("Error: need 3 cubes but only found ", len(cubes), " cube(s)")
    else:
        robot.say_text("I found all the cubes").wait_for_completed()

    for cube in cubes:
        print(f"Cube id:{cube.cube_id}")
        print(f"Postion (x,y,z):{cube.pose.position.x, cube.pose.position.y, cube.pose.position.z}")
        print(f"Postion (x_y_z):{cube.pose.position.x_y_z}")
        robot.go_to_object(cube, distance_mm(40)).wait_for_completed()

        cube.object_type
        

cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)
