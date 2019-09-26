import cozmo

#fonction animation prédéfinei zombie
def zombie(robot: cozmo.robot.Robot):
    
    robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabZombie, ignore_body_track=True).wait_for_completed()
 

#cozmo.run_program(zombie)

