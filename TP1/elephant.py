import cozmo

# fonction Cozmo immite l'éléphant avec animation prédéfinie
def elephant(robot: cozmo.robot.Robot):
    robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabElephant).wait_for_completed()


#cozmo.run_program(elephant)
  