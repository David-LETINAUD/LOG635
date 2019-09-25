# -*- coding: utf-8 -*
import time
import sys

import cozmo
from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes, ObservableElement, ObservableObject
from cozmo.util import Pose,degrees,distance_mm
from avoid_collision import custom_object_pose


from coffee import *
from camera import *
from alarm import *
from reveil import *
from cube_stack import *
from cubes_unstack import *
from nap import *
from sing import *
from mirror import *
from boo import *
from elephant import *
from text import *
from known_face import *
from lastone import *
from roll import *
from zombie import *
custom_object = None

# list FIFO
ID_path = [0,1,2,3,4,5,6,7,8,9,10,11,12]#,13,14,15]
Function_path = [reveil,alarm_clock, coffee,mirror,sing, nap,known_face,zombie,boo,elephant, text,lastone,cube_roll,cube_stack,cube_unstack ]

marker = []
marker_id = []
pose_tab = []
obj_tab = []

max_cust_obj = 4

def handle_object_appeared(evt, **kw):   
    global marker 
    global marker_id
    global pose_tab
    global obj_tab

    # Cela sera appelé chaque fois qu'un EvtObjectAppeared est déclenché
    # chaque fois qu'un objet entre en vue
    if isinstance(evt.obj, CustomObject):
        type_nb = int(str(str(evt.obj.object_type).split('.')[1])[-2:])
        print(type_nb)
        print(f"Cozmo started seeing a {str(evt.obj.object_type) } " +  str(str(evt.obj.object_type).split('.')[1])[-2:] + " ID " + str(evt.obj.object_id)  )
        #object_found.append(evt.obj.get_id())
        if type_nb not in marker_id:
            marker_id.append(type_nb)
            obj_tab.append(evt.obj)
            #pose_tab.append(Pose(evt.obj.pose.position.x -0, evt.obj.pose.position.y - 0, 0, angle_z= degrees(0)))
        else :
            ind = marker_id.index(type_nb)
            print("Position mise à jour")
            # actualiser la position
            obj_tab[ind] = evt.obj
            #pose_tab[ind] = Pose(evt.obj.pose.position.x -0, evt.obj.pose.position.y - 0, 0, angle_z= degrees(0))

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

    global marker 
    global marker_id
    global pose_tab
    global obj_tab

    path_object = ['robot.world.define_custom_cube(CustomObjectTypes.CustomType00,CustomObjectMarkers.Circles2,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType01,CustomObjectMarkers.Circles3,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType02,CustomObjectMarkers.Circles4,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType03,CustomObjectMarkers.Triangles3,60, 24.19, 24.19, True)',
                   #'robot.world.define_custom_cube(CustomObjectTypes.CustomType03,CustomObjectMarkers.Circles5,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType04,CustomObjectMarkers.Diamonds2,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType05,CustomObjectMarkers.Diamonds3,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType06,CustomObjectMarkers.Diamonds4,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType07,CustomObjectMarkers.Diamonds5,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType08,CustomObjectMarkers.Hexagons2,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType09,CustomObjectMarkers.Hexagons3,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType10,CustomObjectMarkers.Hexagons4,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType11,CustomObjectMarkers.Hexagons5,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType12,CustomObjectMarkers.Triangles2,60, 24.19, 24.19, True)'#,
                #    'robot.world.define_custom_cube(CustomObjectTypes.CustomType13,CustomObjectMarkers.Triangles3,60, 24.19, 24.19, True)',
                #    'robot.world.define_custom_cube(CustomObjectTypes.CustomType14,CustomObjectMarkers.Triangles4,60, 24.19, 24.19, True)',
                #    'robot.world.define_custom_cube(CustomObjectTypes.CustomType15,CustomObjectMarkers.Triangles5,60, 24.19, 24.19, True)'
    ]

    for cust_cube in path_object[0:max_cust_obj]:
        eval(cust_cube)
    
    if (path_object is not None):# and  path_object[1] is not None):
        print("All objects defined successfully!")
    else:
        print("One or more object definitions failed!")
        return

    initial_pose = robot.pose
    id_prec = 0
    while len(ID_path)!=0 :
        print("WHILE")
        num_cust_obj = 1
        robot.turn_in_place(degrees(-17*id_prec)).wait_for_completed()
        lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
        print("look_end")

        marker = robot.world.wait_until_observe_num_objects(num=num_cust_obj, object_type=CustomObject, timeout=10)
        print("marker_end")
        #marker = robot.world.wait_until_observe_num_objects(num=2, object_type=cozmo.objects.LightCube, timeout=60)
        lookaround.stop()
        
        while ID_path[0] in marker_id:
            print("FIND")
            print(marker_id , ID_path[0])
            cible = marker_id.index(ID_path[0])
            print(cible)#, obj_tab[cible].object_type)

            

            collision_avoid_pose = custom_object_pose(robot,obj_tab[cible] )
            #robot.go_to_pose(pose_tab[cible], relative_to_robot=False).wait_for_completed()
            robot.go_to_pose(collision_avoid_pose, relative_to_robot=False).wait_for_completed()
            
            robot.add_event_handler(cozmo.world.EvtNewCameraImage, on_new_camera_image)        
            take_photo(robot)

            print("picture ok")
            Function_path[0](robot)
            print("function ok")

            robot.go_to_pose(initial_pose, relative_to_robot=False).wait_for_completed()            
            id_prec = ID_path[0]
            ID_path.pop(0) # POP le 1er élément
            Function_path.pop(0) 
            #obj_tab.pop(0)
            
            path_object.pop(0) # ne pas redétecter les objets sur lequel on est déjà passé
            robot.world.undefine_all_custom_marker_objects()
            for cust_cube in path_object[0:min(max_cust_obj, len(path_object))]:
                eval(cust_cube)
            #robot.drive_straight(distance_mm(-250), speed_mmps(50)).wait_for_completed()
            
            print(ID_path)
            if len(ID_path)==0:
                break 

    # finir par les fonctions cubes
    while len(Function_path)!=0 :    
        Function_path[0](robot)
        print("function ok")
        Function_path.pop(0) 


# Indiquer le dossier pour stocker les photos
global directory    
directory = f"{strftime('%y%m%d')}"
if not os.path.exists('photos'):
    os.makedirs('photos')

#cozmo.run_program(alarm_clock)
cozmo.run_program(custom_objects, use_3d_viewer=True, use_viewer=True, force_viewer_on_top=True)