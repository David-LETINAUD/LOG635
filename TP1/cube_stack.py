import cozmo

def cube_stack(robot: cozmo.robot.Robot):
    # Essai d'émpiler 2 cubes

    lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    cubes = robot.world.wait_until_observe_num_objects(num=2, object_type=cozmo.objects.LightCube, timeout=60)
    lookaround.stop()


    if len(cubes) < 2:
        print("Error: need 2 Cubes but only found", len(cubes), "Cube(s)")
    else:
        # Essai de ramasser le 1er cube
        current_action = robot.pickup_object(cubes[0], num_retries=3)
        current_action.wait_for_completed()
        if current_action.has_failed:
            code, reason = current_action.failure_reason
            result = current_action.result
            print("Pickup Cube failed: code=%s reason='%s' result=%s" % (code, reason, result))
            return

        # Maintenant, essai de placer ce cube sur le 2ème.
        current_action = robot.place_on_object(cubes[1], num_retries=3)
        current_action.wait_for_completed()
        if current_action.has_failed:
            code, reason = current_action.failure_reason
            result = current_action.result
            print("Place On Cube failed: code=%s reason='%s' result=%s" % (code, reason, result))
            return

        print("Cozmo successfully stacked 2 blocks!")

cozmo.run_program(cube_stack, use_3d_viewer=True, use_viewer=True)
