import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("Cannot import numpy: Do `pip3 install --user numpy` to install")

try:
    from PIL import Image
except ImportError:
    sys.exit("Cannot import from PIL: Do `pip3 install --user Pillow` to install")

import cozmo

# parametrer position tete et bras cozmo pour voir l'écran
def get_in_position(robot: cozmo.robot.Robot):
    '''If necessary, Move Cozmo's Head and Lift to make it easy to see Cozmo's face.'''
    if (robot.lift_height.distance_mm > 45) or (robot.head_angle.degrees < 40):
        with robot.perform_off_charger():
            lift_action = robot.set_lift_height(0.0, in_parallel=True)
            head_action = robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE,
                                               in_parallel=True)
            lift_action.wait_for_completed()
            head_action.wait_for_completed()

#seuillage image
def calc_pixel_threshold(image: Image):
    '''Calculate a pixel threshold based on the image.

    Anything brighter than this will be shown on (light blue).
    Anything darker will be shown off (black).
    '''

    # Convertit en image en niveau de gris
    grayscale_image = image.convert('L')

    # calcul de la valeur moyenne
    mean_value = np.mean(grayscale_image.getdata())
    return mean_value

#focntion affichage de la camera sur l'écran
def mirror(robot: cozmo.robot.Robot):
    '''Continuously display Cozmo's camera feed back on his face.'''

    robot.camera.image_stream_enabled = True #accès a la camera
    get_in_position(robot)

    face_dimensions = cozmo.oled_face.SCREEN_WIDTH, cozmo.oled_face.SCREEN_HALF_HEIGHT
    duration_s = 0.1  # time to display each camera frame on Cozmo's face
    cpt = 0
    while cpt<70:
        latest_image = robot.world.latest_image     #dernière image pour 'live'

        if latest_image is not None:
            # redimmensionner l'image pour l'afficher sur l'écran
            resized_image = latest_image.raw_image.resize(face_dimensions,
                                                          Image.BICUBIC)

            # inverser les côtés gauche et droite pour l'effet miroir
            resized_image = resized_image.transpose(Image.FLIP_LEFT_RIGHT)

            # Calculate the pixel threshold for this image. This threshold
            # will define how bright a pixel needs to be in the source image
            # for it to be displayed as lit-up on Cozmo's face.
            pixel_threshold = calc_pixel_threshold(resized_image)

            # Convertir l'image au format pour l'afficher sur l'écran
            screen_data = cozmo.oled_face.convert_image_to_screen_data(
                resized_image,
                pixel_threshold=pixel_threshold)

            # afficher l'image sur l'écran
            robot.display_oled_face_image(screen_data, duration_s * 1000.0)

        time.sleep(duration_s)
        cpt = cpt + 1
    time.sleep(0.5)

#cozmo.run_program(cozmo_face_mirror)
