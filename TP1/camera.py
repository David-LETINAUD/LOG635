# -*- coding: utf-8 -*
import time
from time import strftime
import datetime
import sys
import os

import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

# GLOBALS
directory = '.'
liveCamera = False

#fonction appellée a chaque photo prise
def on_new_camera_image(evt, **kwargs):    
    global liveCamera
    if liveCamera:
        print("Cozmo is taking a photo")
        pilImage = kwargs['image'].raw_image #formatage image
        global directory
        pilImage.save(f"photos/{directory}/{directory}-{kwargs['image'].image_number}.jpeg", "JPEG") #sauvegarde l'image dans le dossier

def take_photo(robot: cozmo.robot.Robot):
    global liveCamera    
    
    # Assurez-vous que la tête et le bras de Cozmo sont à un niveau raisonnable
    #robot.set_head_angle(degrees(10.0)).wait_for_completed()
    robot.set_lift_height(0.0).wait_for_completed()
        
    liveCamera = True #prend photo
    time.sleep(0.1)    
    liveCamera = False   
