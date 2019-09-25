import cozmo
import time

# Récupérer le nom d'un visage reconnu
def known_face(robot: cozmo.robot.Robot):
    
    findfaces= robot.start_behavior(cozmo.behavior.BehaviorTypes.FindFaces)    
    face = robot.world.wait_for_observed_face(timeout=None, include_existing=True)
    findfaces.stop()
    
    if face is not None:
        robot.say_text(f"{face.name}").wait_for_completed()
    
#cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)