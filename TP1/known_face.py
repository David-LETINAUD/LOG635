import cozmo
import time

# Récupérer le nom d'un visage reconnu
def known_face(robot: cozmo.robot.Robot):
    
    findfaces= robot.start_behavior(cozmo.behavior.BehaviorTypes.FindFaces)    #cherche un visage
    face = robot.world.wait_for_observed_face(timeout=None, include_existing=True)  #associe le visage s'il le connait
    findfaces.stop()   #arrete de chercher
    
    if face is not None: #s'il a reconnu un visage dit son nom
        robot.say_text(f"{face.name}").wait_for_completed()
        time.sleep(2)
    