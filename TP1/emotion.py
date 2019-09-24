import cozmo


def emotion(robot: cozmo.robot.Robot):
    print("Exécution de la trigger : OnSpeedtapGameCozmoWinHighIntensity (sans body track)")
    robot.play_anim_trigger(cozmo.anim.Triggers.OnSpeedtapGameCozmoWinHighIntensity, ignore_body_track=True).wait_for_completed()
    robot.say_text("et encore une", duration_scalar=0.5).wait_for_completed()

#cozmo.run_program(emotion)

# # Jouer une animation  en ignorant le mouvement des roues
