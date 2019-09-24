# -*- coding: utf-8 -*
import time
import sys

import cozmo
from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes, ObservableElement, ObservableObject
from cozmo.util import Pose,degrees,distance_mm

from coffee import *
from camera import *
from alarm import *
from reveil import reveil
from cube_stack import *

custom_object = None

# list FIFO
ID_path = [4,5,6,7,8,9,10,11,12,13,14,15]
Function_path = [coffee, reveil, alarm_clock]



# object found
object_found = [] 

def handle_object_appeared(evt, **kw):   
    # Cela sera appelé chaque fois qu'un EvtObjectAppeared est déclanché
    # chaque fois qu'un objet entre en vue
    if isinstance(evt.obj, CustomObject):
        print(f"Cozmo started seeing a {str(evt.obj.object_type)}")
        #object_found.append(evt.obj.get_id())
        #if ID_path[0] in res:
            # go to pose #aller à la position de ID_path[0]
            #ID_path.pop(0) # POP le 1er élément
            #print(ID_path)

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


    path_object = ['robot.world.define_custom_cube(CustomObjectTypes.CustomType00,CustomObjectMarkers.Circles2,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType01,CustomObjectMarkers.Circles3,60, 24.19, 24.19, True)'
    ]

    #path_object.pop()         
    #robot.world.undefine_all_custom_marker_objects()
    for cust_cube in path_object:
        eval(cust_cube)
    

    #for fnc in path_object:
    #    fnc()

    if (path_object is not None):# and  path_object[1] is not None):
        print("All objects defined successfully!")
    else:
        print("One or more object definitions failed!")
        return

    ### 1st step
    #cozmo.run_program(cube_stack, use_3d_viewer=True, use_viewer=True)
    # A TESTER !!! /!\
    
    while len(ID_path)!=0 :
        print("WHILE")
        num_cust_obj = 2
        marker = []
        marker_id = []
        pose_tab = []


        lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
        print("look_end")

        marker = robot.world.wait_until_observe_num_objects(num=num_cust_obj, object_type=CustomObject, timeout=60)
        print("marker_end")
        #marker = robot.world.wait_until_observe_num_objects(num=2, object_type=cozmo.objects.LightCube, timeout=60)
        lookaround.stop()

        #print(marker)
        
        for m in marker:
            print("ID")
            print(m.object_id)
            marker_id.append(m.object_id)
            pose_tab.append(Pose(m.pose.position.x - 90, m.pose.position.y - 0, 0, angle_z= degrees(0)))

        for i in range(num_cust_obj):
            if ID_path[0] in marker_id:
                print("FIND")
                # /!\ selectionner la pose de ID_path[0]
                #print(marker.index(ID_path[0]))
                print(marker_id , ID_path[0])
                cible = marker_id.index(ID_path[0])
                print(cible)
                robot.go_to_pose(pose_tab[cible], relative_to_robot=False).wait_for_completed()
                robot.add_event_handler(cozmo.world.EvtNewCameraImage, on_new_camera_image)        
                take_photo(robot)
                print("picture ok")
                Function_path[0](robot)
                print("function ok")
                ID_path.pop(0) # POP le 1er élément
                Function_path.pop(0) 
                
                path_object.pop(0) # ne pas redétecter les objets sur lequel on est déjà passé
                robot.world.undefine_all_custom_marker_objects()
                for cust_cube in path_object:
                    eval(cust_cube)
                    
                print(ID_path)
                print()
                print()
        

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

#cozmo.run_program(alarm_clock)
cozmo.run_program(custom_objects, use_3d_viewer=True, use_viewer=True, force_viewer_on_top=True)