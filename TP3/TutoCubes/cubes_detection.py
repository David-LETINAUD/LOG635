import cozmo

def cozmo_program(robot: cozmo.robot.Robot):
    lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    cubes = robot.world.wait_until_observe_num_objects(num=3, object_type=cozmo.objects.LightCube, timeout=60)
    lookaround.stop()

    if len(cubes) < 2:
        print("Error: need 3 cubes but only found ", len(cubes), " cube(s)")
    else:
        robot.say_text("I found all the cubes").wait_for_completed()

cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)