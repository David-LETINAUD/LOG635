import cozmo
from cozmo.util import degrees, Pose
WALL_HEIGHT = 50
WALL_WIDTH = 10
def create_walls(robot: cozmo.robot.Robot):
    # --- HORIZONTAL ---
    wall1 = Pose(190, 125, 0, angle_z=degrees(0))
    wall1 = robot.world.create_custom_fixed_object(wall1, WALL_WIDTH, 250, WALL_HEIGHT,relative_to_robot=False)
    wall2 = Pose(90, 125, 0, angle_z=degrees(0))
    wall2 = robot.world.create_custom_fixed_object(wall2, WALL_WIDTH, 100, WALL_HEIGHT,relative_to_robot=False)
    # --- VERTICAL ---
    # wall10 = Pose(190, 500, 0, angle_z=degrees(0))
    # wall10 = robot.world.create_custom_fixed_object(wall10, 380, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    # wall11 = Pose(191, 310, 0, angle_z=degrees(0))
    # wall11 = robot.world.create_custom_fixed_object(wall11, 23, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    wall10 = Pose(280, 250, 0, angle_z=degrees(0))
    wall10 = robot.world.create_custom_fixed_object(wall10, 500, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    wall11 = Pose(250, 251, 0, angle_z=degrees(0))
    wall11 = robot.world.create_custom_fixed_object(wall11, 230, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    wall12 = Pose(120, 124, 0, angle_z=degrees(0))
    wall12 = robot.world.create_custom_fixed_object(wall12, 249, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    wall13 = Pose(0, 253, 0, angle_z=degrees(0))
    wall13 = robot.world.create_custom_fixed_object(wall13, 190, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    # wall14 = Pose(0, 253, 0, angle_z=degrees(0))
    # wall14 = robot.world.create_custom_fixed_object(wall14, 190, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
   
    # wall3 = Pose(140, 125, 0, angle_z=degrees(0))
    # wall3 = robot.world.create_custom_fixed_object(wall3, 100, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    # wall4 = Pose(45, 170, 0, angle_z=degrees(0))
    # wall4 = robot.world.create_custom_fixed_object(wall4, 90, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)

stop_a = Pose(140, 190, 0, angle_z=degrees(-90))
stop_b = Pose(45, 125, 0, angle_z=degrees(0))
stop_c = Pose(140, 60, 0, angle_z=degrees(90))
stop_o = Pose(0, 0, 0, angle_z=degrees(180))

def cozmo_program(robot: cozmo.robot.Robot):
    robot.world.delete_all_custom_objects()
    print(robot.pose.position)
    create_walls(robot)
    robot.go_to_pose(stop_a, relative_to_robot=False).wait_for_completed()
    robot.go_to_pose(stop_b, relative_to_robot=False).wait_for_completed()
    robot.go_to_pose(stop_c, relative_to_robot=False).wait_for_completed()
    robot.go_to_pose(stop_o, relative_to_robot=False).wait_for_completed()

cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)