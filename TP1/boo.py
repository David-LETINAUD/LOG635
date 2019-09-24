import cozmo


def boo(robot: cozmo.robot.Robot):
    robot.play_anim_trigger(cozmo.anim.Triggers.PeekABooSurprised).wait_for_completed()


#cozmo.run_program(boo)
  