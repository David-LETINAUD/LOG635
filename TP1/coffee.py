import cozmo

def coffee(robot: cozmo.robot.Robot):
    robot.say_text("Hy need coffee!", True, in_parallel=True, duration_scalar=2,use_cozmo_voice=True).wait_for_completed()



#cozmo.run_program(coffee)