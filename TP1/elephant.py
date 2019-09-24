import cozmo


def roll(robot: cozmo.robot.Robot):
    robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabElephant).wait_for_completed()


#cozmo.run_program(reveil)
  