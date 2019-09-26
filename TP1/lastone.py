import cozmo


def lastone(robot: cozmo.robot.Robot):
    #dit qu'il a finit
    robot.say_text("Yes, hy finish it!", True, in_parallel=True, duration_scalar=0.5,use_cozmo_voice=True).wait_for_completed() 
    #animation prédéfinie
    robot.play_anim_trigger(cozmo.anim.Triggers.WorkoutPutDown_highEnergy).wait_for_completed()


#cozmo.run_program(lastone)