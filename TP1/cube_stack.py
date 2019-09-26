import cozmo
import time
def cube_stack(robot: cozmo.robot.Robot):
    # Attempt to stack 2 cubes

    # Cozmo regarde autour de lui jusqu'à ce qu'il découvre au moins 2 cubes
    lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    cubes = robot.world.wait_until_observe_num_objects(num=2, object_type=cozmo.objects.LightCube, timeout=60)
    lookaround.stop() #arrete de chercher des cubes

    if len(cubes) < 2: #si il a trouvé moins de 2 cubes affiche erreur
        print("Error: need 2 Cubes but only found", len(cubes), "Cube(s)")
    else:
        # essaye de prendre le premier cube
        current_action = robot.pickup_object(cubes[0], num_retries=5)
        current_action.wait_for_completed()
        if current_action.has_failed: #S'il echoue renvoie erreur
            code, reason = current_action.failure_reason
            result = current_action.result
            print("Pickup Cube failed: code=%s reason='%s' result=%s" % (code, reason, result))
            return

        # essaye de placer ce cube sur le deuxième
        current_action = robot.place_on_object(cubes[1], num_retries=5)
        current_action.wait_for_completed()
        if current_action.has_failed:#s'il echoue renvoie erreur
            code, reason = current_action.failure_reason
            result = current_action.result
            print("Place On Cube failed: code=%s reason='%s' result=%s" % (code, reason, result))
            return

        print("Cozmo successfully stacked 2 blocks!")
    time.sleep(1)