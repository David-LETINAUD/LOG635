# -*- coding: utf-8 -*
import time
import sys

import cozmo
from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes, ObservableElement, ObservableObject
from cozmo.util import Pose,degrees,distance_mm

from camera import *
from alarm import *

custom_object = None

def handle_object_appeared(evt, **kw):   
    # Cela sera appelé chaque fois qu'un EvtObjectAppeared est déclanché
    # chaque fois qu'un objet entre en vue
    if isinstance(evt.obj, CustomObject):
        print(f"Cozmo started seeing a {str(evt.obj.object_type)}")

def handle_object_disappeared(evt, **kw):
    # Cela sera appelé lorsqu'un EvtObjectDisappeared est declanché    
    # chaque fois qu'un objet est hors de vue.
    if isinstance(evt.obj, CustomObject):
        print(f"Cozmo stopped seeing a {str(evt.obj.object_type)}")

def custom_objects(robot: cozmo.robot.Robot):
    # Gestionnaires d'évennements à chaque fois que Cozmo
    # vois ou arrète de voir un objet
    robot.add_event_handler(cozmo.objects.EvtObjectAppeared, handle_object_appeared)
    robot.add_event_handler(cozmo.objects.EvtObjectDisappeared, handle_object_disappeared)

    path_object = [robot.world.define_custom_cube(CustomObjectTypes.CustomType00,
                                                 CustomObjectMarkers.Circles2,
                                                 60, 24.19, 24.19, True),
                   robot.world.define_custom_cube(CustomObjectTypes.CustomType01,
                                                 CustomObjectMarkers.Circles3,
                                                 60, 24.19, 24.19, True) 
                  ]
    
    #print(path_object)
    if (path_object is not None):# and  path_object[1] is not None):
        print("All objects defined successfully!")
    else:
        print("One or more object definitions failed!")
        return

    print("Press CTRL-C to quit")
    
    

    lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    cubes = robot.world.wait_until_observe_num_objects(num=2, object_type=cozmo.objects.LightCube, timeout=60)
    lookaround.stop()

    if len(cubes) < 2:
        print("Error: need 2 Cubes but only found", len(cubes), "Cube(s)")
    else:
        # Try and pickup the 1st cube
        current_action = robot.pickup_object(cubes[0], num_retries=3)
        current_action.wait_for_completed()
        if current_action.has_failed:
            code, reason = current_action.failure_reason
            result = current_action.result
            print("Pickup Cube failed: code=%s reason='%s' result=%s" % (code, reason, result))
            return

        # Now try to place that cube on the 2nd one
        current_action = robot.place_on_object(cubes[1], num_retries=3)
        current_action.wait_for_completed()
        if current_action.has_failed:
            code, reason = current_action.failure_reason
            result = current_action.result
            print("Place On Cube failed: code=%s reason='%s' result=%s" % (code, reason, result))
            return

        print("Cozmo successfully stacked 2 blocks!")

    lookaround1 = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    marker = robot.world.wait_until_observe_num_objects(num=len(path_object), object_type=CustomObject, timeout=120)
    lookaround1.stop()


    if len(marker) > len(path_object)-1:
        print("Found object")
        #a = super(CustomObject, cubes[0])
        #b = super(CustomObject, cubes[1])
        pose_tab = []
        for c in marker:
            pose_tab.append(Pose(c.pose.position.x - 90, c.pose.position.y - 0, 0, angle_z= degrees(0)))

        robot.go_to_pose(pose_tab[0], relative_to_robot=False).wait_for_completed()
        robot.add_event_handler(cozmo.world.EvtNewCameraImage, on_new_camera_image)        
        take_photo(robot)
        robot.say_text("I got it !").wait_for_completed()

        robot.go_to_pose(pose_tab[1], relative_to_robot=False).wait_for_completed()
        robot.add_event_handler(cozmo.world.EvtNewCameraImage, on_new_camera_image)        
        take_photo(robot)


        #robot.go_to_object(cubes[0],distance_mm(50.0)).wait_for_completed()
        #robot.go_to_pose(cubes[1].pose,relative_to_robot=False)
        print("Got to object")       
    else:
        print("Cannot locate custom box")

    while True:
        time.sleep(0.1)


# Indiquer le dossier pour stocker les photos
global directory    
directory = f"{strftime('%y%m%d')}"
if not os.path.exists('photos'):
    os.makedirs('photos')

cozmo.run_program(alarm_clock)
cozmo.run_program(custom_objects, use_3d_viewer=True, use_viewer=True, force_viewer_on_top=True)