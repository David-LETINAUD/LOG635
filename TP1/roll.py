import cozmo

def cube_roll(robot: cozmo.robot.Robot):
    # Essai d'émpiler 2 cubes

    lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    cubes = robot.world.wait_until_observe_num_objects(num=1, object_type=cozmo.objects.LightCube, timeout=60)
    lookaround.stop()


    if len(cubes) <1:
        print("Error: need a Cubes but only found", len(cubes), "Cube(s)")
    else:
        # Essai de faire rouler le cube
        robot.run_timed_behavior(cozmo.behavior.BehaviorTypes.RollBlock, active_time=60)#.wait_for_completed()
        #current_action.wait_for_completed()
        # if current_action.has_failed:
        #     code, reason = current_action.failure_reason
        #     result = current_action.result
        #     print("Pickup Cube failed: code=%s reason='%s' result=%s" % (code, reason, result))
        #     return

        print("Cozmo successfully roll a block!")

#cozmo.run_program(cube_roll, use_3d_viewer=True, use_viewer=True)


