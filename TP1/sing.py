import cozmo


def sing(robot: cozmo.robot.Robot):
    
    robot.play_anim_trigger(cozmo.anim.Triggers.Singing_120bpm).wait_for_completed()

#cozmo.run_program(sing)