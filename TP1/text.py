import datetime
import math
import sys
import time

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Cannot import from PIL. Do `pip3 install --user Pillow` to install")

import cozmo


def make_text(text_to_draw, x, y, font=None):

    # make a blank image for the text, initialized to opaque black
    text_image = Image.new('RGBA', cozmo.oled_face.dimensions(), (0, 0, 0, 255))

    # get a drawing context
    dc = ImageDraw.Draw(text_image)

    # draw the text
    dc.text((x, y), text_to_draw, fill=(255, 255, 255, 255), font=font)

    return text_image


# get a font - location depends on OS so try a couple of options
# failing that the default of None will just use a default font
_clock_font = None
try:
    _clock_font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    try:
        _clock_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 20)
    except IOError:
        pass


def text(robot: cozmo.robot.Robot):
    time_text = "TP1 LOG635"
    clock_image = make_text(time_text, 8, 6, _clock_font)
    oled_face_data = cozmo.oled_face.convert_image_to_screen_data(clock_image)

    # display for 4 seconds
    robot.display_oled_face_image(oled_face_data, 4000.0,in_parallel=True).wait_for_completed()

    # only sleep for a fraction of a second to ensure we update the seconds as soon as they change
    time.sleep(0.1)
