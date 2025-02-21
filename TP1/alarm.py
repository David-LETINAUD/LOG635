import datetime
import math
import sys
import time

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Cannot import from PIL. Do `pip3 install --user Pillow` to install")

import cozmo


#: bool: Set to True to display the clock as analog
#: (with a small digital readout below)
SHOW_ANALOG_CLOCK = False


def make_text_image(text_to_draw, x, y, font=None):
    '''Make a PIL.Image with the given text printed on it

    Args:
        text_to_draw (string): the text to draw to the image
        x (int): x pixel location
        y (int): y pixel location
        font (PIL.ImageFont): the font to use

    Returns:
        :class:(`PIL.Image.Image`): a PIL image with the text drawn on it
    '''

    # make a blank image for the text, initialized to opaque black
    text_image = Image.new('RGBA', cozmo.oled_face.dimensions(), (0, 0, 0, 255))

    # get a drawing context
    dc = ImageDraw.Draw(text_image)

    # dessine le texte
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


def draw_clock_hand(dc, cen_x, cen_y, circle_ratio, hand_length):
    '''Draw a single clock hand (hours, minutes or seconds)

    Args:
        dc: (:class:`PIL.ImageDraw.ImageDraw`): drawing context to use
        cen_x (float): x coordinate of center of hand
        cen_y (float): y coordinate of center of hand
        circle_ratio (float): ratio (from 0.0 to 1.0) that hand has travelled
        hand_length (float): the length of the hand
    '''

    hand_angle = circle_ratio * math.pi * 2.0
    vec_x = hand_length * math.sin(hand_angle)
    vec_y = -hand_length * math.cos(hand_angle)

    # x_scalar doubles the x size to compensate for the interlacing
    # in y that would otherwise make the screen appear 2x tall
    x_scalar = 2.0

    # pointy end of hand
    hand_end_x = int(cen_x + (x_scalar * vec_x))
    hand_end_y = int(cen_y + vec_y)

    # 2 points, perpendicular to the direction of the hand,
    # to give a triangle with some width
    hand_width_ratio = 0.1
    hand_end_x2 = int(cen_x - ((x_scalar * vec_y) * hand_width_ratio))
    hand_end_y2 = int(cen_y + (vec_x * hand_width_ratio))
    hand_end_x3 = int(cen_x + ((x_scalar * vec_y) * hand_width_ratio))
    hand_end_y3 = int(cen_y - (vec_x * hand_width_ratio))

    dc.polygon([(hand_end_x, hand_end_y), (hand_end_x2, hand_end_y2),
                (hand_end_x3, hand_end_y3)], fill=(255, 255, 255, 255))

#fonction paramétrage de l'image à afficher
def make_clock_image(current_time):
    '''Make a PIL.Image with the current time displayed on it

    Args:
        text_to_draw (:class:`datetime.time`): the time to display

    Returns:
        :class:(`PIL.Image.Image`): a PIL image with the time displayed on it
    '''

    time_text = time.strftime("%I:%M:%S %p")

    if not SHOW_ANALOG_CLOCK:
        return make_text_image(time_text, 8, 6, _clock_font)

    # make a blank image for the text, initialized to opaque black
    clock_image = Image.new('RGBA', cozmo.oled_face.dimensions(), (0, 0, 0, 255))

    # get a drawing context
    dc = ImageDraw.Draw(clock_image)

    # calul de la position de l'heure sur l'écran
    text_height = 9
    screen_width, screen_height = cozmo.oled_face.dimensions()
    analog_width = screen_width
    analog_height = screen_height - text_height
    cen_x = analog_width * 0.5
    cen_y = analog_height * 0.5

    # calculate size of clock hands
    sec_hand_length = (analog_width if (analog_width < analog_height) else analog_height) * 0.5
    min_hand_length = 0.85 * sec_hand_length
    hour_hand_length = 0.7 * sec_hand_length

    # calculate rotation for each hand
    sec_ratio = current_time.second / 60.0
    min_ratio = (current_time.minute + sec_ratio) / 60.0
    hour_ratio = (current_time.hour + min_ratio) / 12.0

    # draw the clock hands
    draw_clock_hand(dc, cen_x, cen_y, hour_ratio, hour_hand_length)
    draw_clock_hand(dc, cen_x, cen_y, min_ratio, min_hand_length)
    draw_clock_hand(dc, cen_x, cen_y, sec_ratio, sec_hand_length)

    # draw the digital time_text at the bottom
    x = 32
    y = screen_height - text_height
    dc.text((x, y), time_text, fill=(255, 255, 255, 255), font=None)

    return clock_image

#fonction pour mettre sous format horaire
def convert_to_time_int(in_value, time_unit):
    '''Convert in_value to an int and ensure it is in the valid range for that time unit

    (e.g. 0..23 for hours)'''

    max_for_time_unit = {'hours': 23, 'minutes': 59, 'seconds': 59}
    max_val = max_for_time_unit[time_unit]

    try:
        int_val = int(in_value)
    except ValueError:
        raise ValueError("%s value '%s' is not an int" % (time_unit, in_value))

    if int_val < 0:
        raise ValueError("%s value %s is negative" % (time_unit, int_val))

    if int_val > max_val:
        raise ValueError("%s value %s exceeded %s" % (time_unit, int_val, max_val))

    return int_val


def extract_time_from_args():
    ''' Extract a (24-hour-clock) user-specified time from the command-line

    Supports colon and space separators - e.g. all 3 of "11 22 33", "11:22:33" and "11 22:33"
    would map to the same time.
    The seconds value is optional and defaults to 0 if not provided.'''

    # split sys.argv further for any args that contain a ":"
    split_time_args = []
    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]
        split_args = arg.split(':')
        for split_arg in split_args:
            split_time_args.append(split_arg)

    if len(split_time_args) >= 2:
        try:
            hours = convert_to_time_int(split_time_args[0], 'hours')
            minutes = convert_to_time_int(split_time_args[1], 'minutes')
            seconds = 0
            if len(split_time_args) >= 3:
                seconds = convert_to_time_int(split_time_args[2], 'seconds')

            return datetime.time(hours, minutes, seconds)
        except ValueError as e:
            print("ValueError %s" % e)

    # Default to no alarm
    return None

#definie position du robot pour mieux voir l'heure
def get_in_position(robot: cozmo.robot.Robot):
    '''If necessary, Move Cozmo's Head and Lift to make it easy to see Cozmo's face'''
    if (robot.lift_height.distance_mm > 45) or (robot.head_angle.degrees < 40):
        with robot.perform_off_charger():
            lift_action = robot.set_lift_height(0.0, in_parallel=True)
            head_action = robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE,
                                               in_parallel=True)
            lift_action.wait_for_completed()
            head_action.wait_for_completed()

#fonction principale
def alarm_clock(robot: cozmo.robot.Robot):
    '''The core of the alarm_clock program'''
    get_in_position(robot)
    
    current_time = datetime.datetime.now().time()

    
    # Create the updated image with this time
    clock_image = make_clock_image(current_time)
    oled_face_data = cozmo.oled_face.convert_image_to_screen_data(clock_image)

    # affichage 4s
    robot.display_oled_face_image(oled_face_data, 4000.0,in_parallel=True).wait_for_completed()

    # only sleep for a fraction of a second to ensure we update the seconds as soon as they change
    time.sleep(0.1)

    short_time_string = str(current_time.hour) + ":" + str(current_time.minute)
    #Cozmo dit wake up and l'heure qu'il est
    robot.say_text("Wake up ! It's " + short_time_string, True, in_parallel=True, duration_scalar=0.5).wait_for_completed()
    #robot.play_anim_trigger(cozmo.anim.Triggers.WorkoutPutDown_highEnergy).wait_for_completed()
