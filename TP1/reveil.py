import cozmo


def reveil(robot: cozmo.robot.Robot):
    robot.play_anim_trigger(cozmo.anim.Triggers.ConnectWakeUp).wait_for_completed()


#cozmo.run_program(reveil)
    