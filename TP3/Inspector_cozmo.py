# -*- coding: utf-8 -*
import time
from time import strftime
import sys
import os

import cozmo
from cozmo.objects import CustomObject, CustomObjectMarkers, CustomObjectTypes, ObservableElement, ObservableObject
from cozmo.objects import LightCube1Id, LightCube2Id, LightCube3Id
from cozmo.util import Pose,degrees,distance_mm, speed_mmps
from avoid_collision import custom_object_pose

from find_the_murderer import *
from Map import create_walls

#FIFO
# Chemin dans l'ordre (numéro correspond au numéro du CustomTypeXX)
Cust_obj_path = [4,5,6]
# Fonctions cubes et fonctions associés aux objets
Function_path = [analyse_victime,indice_1,indice_2,indice_3]

# stocke les objets détectés
Obj_detect = []
# stock les numéros des CustomTypeXX des objets détéctés
Cust_type_detect = []

def handle_object_appeared(evt, **kw):   
    global Cust_type_detect
    global Obj_detect

    # Cela sera appelé chaque fois qu'un EvtObjectAppeared est déclenché
    # chaque fois qu'un objet entre en vue
    if isinstance(evt.obj, CustomObject):
        # Récupération du numéro du CustomTypeXX
        type_nb = int(str(str(evt.obj.object_type).split('.')[1])[-2:])
        print(type_nb)
        print(f"Cozmo started seeing a {str(evt.obj.object_type) } " +  str(str(evt.obj.object_type).split('.')[1])[-2:] + " ID " + str(evt.obj.object_id)  )
        
        # Si détecté pour la 1ère fois, l'enregistrement dans Obj_detect et Cust_type_detect  
        if type_nb not in Cust_type_detect:
            Cust_type_detect.append(type_nb)
            Obj_detect.append(evt.obj)
        # Sinon mise à jour de Obj_detect (permet de mettre à jour l'objet et nottament sa position)
        else :
            ind = Cust_type_detect.index(type_nb)
            print("Position mise à jour")
            # actualiser la position
            Obj_detect[ind] = evt.obj

def handle_object_disappeared(evt, **kw):
    # Cela sera appelé lorsqu'un EvtObjectDisappeared est declanché    
    # chaque fois qu'un objet est hors de vue.
    if isinstance(evt.obj, CustomObject):
        print(f"Cozmo stopped seeing a {str(evt.obj.object_type)}")

def custom_objects(robot: cozmo.robot.Robot):
    # Gestionnaires d'évennements à chaque fois que Cozmo..
    # ..vois ou arrète de voir un objet    
    robot.add_event_handler(cozmo.objects.EvtObjectAppeared, handle_object_appeared)
    robot.add_event_handler(cozmo.objects.EvtObjectDisappeared, handle_object_disappeared)
    # Cube2 taped
    robot.add_event_handler(cozmo.objects.EvtObjectTapped, handle_object_tapped)

    create_walls(robot)

    global Cust_type_detect
    global Obj_detect

    path_object = ['robot.world.define_custom_cube(CustomObjectTypes.CustomType04,CustomObjectMarkers.Diamonds2,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType05,CustomObjectMarkers.Diamonds3,60, 24.19, 24.19, True)',
                   'robot.world.define_custom_cube(CustomObjectTypes.CustomType06,CustomObjectMarkers.Diamonds4,60, 24.19, 24.19, True)'
                    ]
    # Sauvegarde de la position initiale du robot
    # trouver une meilleur position => la position centrale de la map
    initial_pose = robot.pose

    # Seul les 4 premiers objets sont détectables
    for cust_cube in path_object: #[0:max_cust_obj]:
        eval(cust_cube)

    lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    cubes = robot.world.wait_until_observe_num_objects(num=2, object_type=cozmo.objects.LightCube, timeout=60)
    lookaround.stop()


    # Recherche les cubes
    # get cube 1
    cube1 = robot.world.get_light_cube(LightCube1Id)
    # get cube 3
    cube3 = robot.world.get_light_cube(LightCube3Id)

    robot.go_to_pose(cube1.pose, relative_to_robot=False).wait_for_completed() 
    #Actions
    Function_path[0](robot)
    Function_path.pop(0)
   
    robot.go_to_pose(initial_pose, relative_to_robot=False).wait_for_completed() 

    # Faire toutes les autres actions associés aux marqueurs
    while len(Cust_obj_path)!=0 :

        # Faire tourner le robot pour mieux détecter les prochains objets
        #robot.turn_in_place(degrees(-17*Cust_obj_path[0])).wait_for_completed()

        lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
        robot.world.wait_until_observe_num_objects(num=1, object_type=CustomObject, timeout=10)
        lookaround.stop()
        
        # Tant que le prochain objet à exécuter est déjà détecté (présent dans Cust_type_detect)
        while Cust_obj_path[0] in Cust_type_detect:
            # Retourne l'index du prochain objet dans Cust_type_detect 
            cible = Cust_type_detect.index(Cust_obj_path[0])
            print(Cust_type_detect , Cust_obj_path[0])

            # Calcul de la position optimale pour éviter les objets
            collision_avoid_pose = custom_object_pose(robot,Obj_detect[cible] )
            # Se diriger vers l'objet
            robot.go_to_pose(collision_avoid_pose, relative_to_robot=False).wait_for_completed()
            
            # Executer la fonction associée à l'objet
            Function_path[0](robot)

            # Retourner à la position initiale 
            robot.go_to_pose(initial_pose, relative_to_robot=False).wait_for_completed()     
            Cust_obj_path.pop(0) # POP le 1er élément des listes FIFO
            Function_path.pop(0)                    
            path_object.pop(0) # Ne pas redétecter les objets sur lequel on est déjà passé

            # Rendre indétectable tous les objets
            #robot.world.undefine_all_custom_marker_objects()
            # Rendre detectable ceux qui reste
            # for cust_cube in path_object:
            #     eval(cust_cube)
            
            # Sortir de la boucle si Cust_obj_path devient vide
            if len(Cust_obj_path)==0:
                break 

    
    robot.go_to_pose(initial_pose, relative_to_robot=False).wait_for_completed() 
    Conclusions()
    robot.say_text("C'est {} qui à tué {}!".format(agent.get_suspect(), agent.get_victim())).wait_for_completed()

    current_action = robot.pickup_object(cube3, num_retries=5)
    current_action.wait_for_completed()
    robot.turn_in_place(degrees(90)).wait_for_completed()
    robot.drive_straight(distance_mm(500), speed_mmps(300)).wait_for_completed()
            
cozmo.run_program(custom_objects, use_3d_viewer=True, use_viewer=True, force_viewer_on_top=True)
