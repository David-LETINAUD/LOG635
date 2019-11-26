import cozmo
from cozmo.util import degrees,Pose

WALL_HEIGHT = 50
WALL_WIDTH = 10

def create_walls(robot: cozmo.robot.Robot):
    # --- HORIZONTAL ---
    wall1 = Pose(380, 250, 0, angle_z=degrees(0))
    wall1 = robot.world.create_custom_fixed_object(wall1, WALL_WIDTH, 500, WALL_HEIGHT,relative_to_robot=False)
    
    wall2 = Pose(0, 320, 0, angle_z=degrees(0))
    wall2 = robot.world.create_custom_fixed_object(wall2, WALL_WIDTH, 360, WALL_HEIGHT,relative_to_robot=False)
    
    wall3 = Pose(200, 440, 0, angle_z=degrees(0))
    wall3 = robot.world.create_custom_fixed_object(wall3, WALL_WIDTH, 120, WALL_HEIGHT,relative_to_robot=False)

    wall4 = Pose(280, 175, 0, angle_z=degrees(0))
    wall4 = robot.world.create_custom_fixed_object(wall4, WALL_WIDTH, 50, WALL_HEIGHT,relative_to_robot=False)
    
    wall5 = Pose(100, 150, 0, angle_z=degrees(0))
    wall5 = robot.world.create_custom_fixed_object(wall5, WALL_WIDTH, 20, WALL_HEIGHT,relative_to_robot=False)
    
    # # --- VERTICAL ---
    wall6 = Pose(50, 140, 0, angle_z=degrees(0))
    wall6 = robot.world.create_custom_fixed_object(wall6, 100, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    
    wall7 = Pose(50, 260, 0, angle_z=degrees(0))
    wall7 = robot.world.create_custom_fixed_object(wall7, 100, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)

    wall8 = Pose(50, 380, 0, angle_z=degrees(0))
    wall8 = robot.world.create_custom_fixed_object(wall8, 100, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)

    wall9 = Pose(190, 500, 0, angle_z=degrees(0))
    wall9 = robot.world.create_custom_fixed_object(wall9, 380, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)

    wall10 = Pose(330, 300, 0, angle_z=degrees(0))
    wall10 = robot.world.create_custom_fixed_object(wall10, 100, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)

    wall11 = Pose(330, 200, 0, angle_z=degrees(0))
    wall11 = robot.world.create_custom_fixed_object(wall11, 100, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)

    wall12 = Pose(260, 150, 0, angle_z=degrees(0))
    wall12 = robot.world.create_custom_fixed_object(wall12, 40, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)

    wall13 = Pose(310, 0, 0, angle_z=degrees(0))
    wall13 = robot.world.create_custom_fixed_object(wall13, 140, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)

stop= []
stop.append(Pose(50, 200, 0, angle_z=degrees(180)))
stop.append(Pose(50, 330, 0, angle_z=degrees(180)))
stop.append(Pose(70, 450, 0, angle_z=degrees(180)))
stop.append(Pose(300, 410, 0, angle_z=degrees(180)))
stop.append(Pose(330, 250, 0, angle_z=degrees(0)))
stop.append(Pose(330, 90, 0, angle_z=degrees(180)))