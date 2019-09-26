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

def on_new_camera_image(evt, **kwargs):    
    global liveCamera
    if liveCamera:
        print("Cozmo is taking a photo")
        pilImage = kwargs['image'].raw_image
        global directory
        pilImage.save(f"photos/{directory}/{directory}-{kwargs['image'].image_number}.jpeg", "JPEG")

def take_photo(robot: cozmo.robot.Robot):
    global liveCamera    
    
    # Assurez-vous que la tête et le bras de Cozmo sont à un niveau raisonnable
    robot.set_head_angle(degrees(3.0)).wait_for_completed()
    robot.set_lift_height(0.0).wait_for_completed()
        
    liveCamera = True
    time.sleep(0.1)    
    liveCamera = False   
