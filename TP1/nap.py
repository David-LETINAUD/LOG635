import cozmo

#fonction Cozmo fait la sieste avec 3 animation prédéfinies 
def nap(robot: cozmo.robot.Robot):
    
    robot.play_anim_trigger(cozmo.anim.Triggers.GoToSleepGetIn).wait_for_completed()    #va dormir
    robot.play_anim_trigger(cozmo.anim.Triggers.Sleeping).wait_for_completed()          #dors
    robot.play_anim_trigger(cozmo.anim.Triggers.GoToSleepGetOut).wait_for_completed()   #se reville


#cozmo.run_program(nap)