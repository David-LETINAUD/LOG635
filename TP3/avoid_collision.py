from math import sqrt

from cozmo.util import Pose

from sympy import Eq, symbols, solve
from numpy import ones,vstack
from numpy.linalg import lstsq

# Symbols utiilsés dans les équations symboliques
x, y = symbols("x y")


def line_equation(p1_x, p1_y, p2_x, p2_y):
    """Cette fonction obtient la pente et l'ordonnée à l'origine
    d'une droite qui passe par deux points connus.

    Dans ce cas, le premier point c'est Cozmo, et le deuxième, c'est l'objet.

    Arguments:
        p1_x {float} -- Coordonée x du premier point (Cozmo)
        p1_y {float} -- Coordonée y du premier point (Cozmo)
        p2_x {float} -- Coordonée x du deuxième point (Objet)
        p2_y {float} -- Coordonée y du deuxième point (Objet)

    Returns:
        (float, float) -- La pente (m) et l'ordonnée à l'origine (b)
    """
    points = [(p1_x, p1_y),(p2_x, p2_y)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, b = lstsq(A, y_coords, rcond=None)[0]
    return m, b


def nearest_intersection(p_x, p_y, points):
    """Cette fonction obtient le point le plus proche (entre deux) à un autre point de référence.

    Arguments:
        p_x {float} -- Coordonée x du point de référence (Cozmo)
        p_y {float} -- Coordonée y du point de référence (Cozmo)
        points {tuple} -- Deux points à tester

    Returns:
        dict -- Le point le plus proche
    """
    dx_1 = (p_x - points[0][x])
    dy_1 = (p_y - points[0][y])
    dist_1 = sqrt(dx_1**2 + dy_1**2)

    dx_2 = (p_x - points[1][x])
    dy_2 = (p_y - points[1][y])
    dist_2 = sqrt(dx_2**2 + dy_2**2)

    if dist_1 < dist_2:
        return points[0]
    return points[1]

def custom_object_pose(robot, custom_object):
    """Cette fonction trace une ligne droite (imaginaire) entre Cozmo et l'objet à atteindre.
    Pour évite un collision, on crée un cercle autour l'objet (imaginaire) et obtient le point
    le plus proche.

              Cozmo                         Objet
                |                             |
                v                             v

                                            *   *
                                        *           *
                                       *             *
               -o---------------------X-------o-------X
                                       *             *
                                        *           *
                                            *   *
                                       ^             ^
                                       |_____________|
                                              |
                                      Point d'intersection,
                                      choisir le plus proche
                                      à cozmo.

    Arguments:
        robot {cozmo.robot.Robot} -- Le robot.
        custom_object {cozmo.object.CustomObject} -- L'object à atteindre.

    Returns:
        Pose -- La point du point d'approximation le plus proche a Cozmo.
    """

    print(">>> robot: ", robot.pose.position)
    print(">>> cube: ", custom_object.pose.position)

    m, b = line_equation(robot.pose.position.x, robot.pose.position.y,
                            custom_object.pose.position.x, custom_object.pose.position.y)
    print(">>> m, b: ", m, b)

    # Équation symbolique de la droite : y = mx + b => y - mx = b
    line = Eq(y - m*x, b)
    print(f">>> line: y = {m} x + {b}")

    # Équation symbolique d'un circle autour de l'objet : (x-c.x)^2 + (y-c.y)^2 = r
    # !!! Le rayon du cercle dépends de la taille de vos objets !!!
    circle = Eq((x - custom_object.pose.position.x)**2 + (y - custom_object.pose.position.y)**2, 20000)
    print(f">>> circle: (x - {custom_object.pose.position.x})**2 + (y - {custom_object.pose.position.y})**2")

    # Points d'intersection entre la droite et le cercle
    intersection_points = solve([line, circle])
    print(">>> intesection points: ", intersection_points)

    # Obtention du point le plus proche à Cozmo
    point = nearest_intersection(robot.pose.position.x, robot.pose.position.y, intersection_points)
    print(">>> nearest intersection: ", point)

    # On retourne la pose qui évite la collision
    return Pose(point[x], point[y], 0, angle_z=custom_object.pose.rotation.angle_z)
