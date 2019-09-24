import cozmo


def nap(robot: cozmo.robot.Robot):
    
    robot.play_anim_trigger(cozmo.anim.Triggers.GoToSleepGetIn).wait_for_completed()
    robot.play_anim_trigger(cozmo.anim.Triggers.Sleeping).wait_for_completed()
    robot.play_anim_trigger(cozmo.anim.Triggers.GoToSleepGetOut).wait_for_completed()


#cozmo.run_program(nap)