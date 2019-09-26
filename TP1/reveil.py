import cozmo

#fonction Cozmo se reveille comportepment à partir d'une animation prédéfinie
def reveil(robot: cozmo.robot.Robot):
    robot.play_anim_trigger(cozmo.anim.Triggers.ConnectWakeUp).wait_for_completed()


#cozmo.run_program(reveil)
    