import cozmo

def lastone(robot: cozmo.robot.Robot):
    robot.say_text("Okay, let's mouve some cubes!", True, in_parallel=True, duration_scalar=0.5,use_cozmo_voice=True).wait_for_completed()
    robot.play_anim_trigger(cozmo.anim.Triggers.WorkoutPutDown_highEnergy).wait_for_completed()


#cozmo.run_program(lastone)