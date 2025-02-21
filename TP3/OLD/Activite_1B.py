import cozmo
from cozmo.util import degrees, Pose
WALL_HEIGHT = 50
WALL_WIDTH = 10
def create_walls(robot: cozmo.robot.Robot):
    # JS
    # --- HORIZONTAL ---
    wall1_h = Pose(0, 250, 0, angle_z=degrees(0))
    wall1_h = robot.world.create_custom_fixed_object(wall1_h, WALL_WIDTH, 260, WALL_HEIGHT,relative_to_robot=False)
    wall2_h = Pose(370, 125, 0, angle_z=degrees(0))
    wall2_h = robot.world.create_custom_fixed_object(wall2_h, WALL_WIDTH, 250, WALL_HEIGHT,relative_to_robot=False)
    wall3_h = Pose(500, 330, 0, angle_z=degrees(0))
    wall3_h = robot.world.create_custom_fixed_object(wall3_h, WALL_WIDTH, 100, WALL_HEIGHT,relative_to_robot=False)
    # --- VERTICAL --
    wall10 = Pose(250, 380, 0, angle_z=degrees(0))
    wall10 = robot.world.create_custom_fixed_object(wall10, 500, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    
    wall11 = Pose(250, 250, 0, angle_z=degrees(0))
    wall11 = robot.world.create_custom_fixed_object(wall11, 230, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    

    # Probleme ici !
    wall12 = Pose(120, 120, 0, angle_z=degrees(0))
    wall12 = robot.world.create_custom_fixed_object(wall12, 240, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    

    # wall12 = Pose(120, 120, 0, angle_z=degrees(0))
    # wall12 = robot.world.create_custom_fixed_object(wall12, 240, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
    
    # wall13 = Pose(0, 253, 0, angle_z=degrees(0))
    # wall13 = robot.world.create_custom_fixed_object(wall13, 190, WALL_WIDTH, WALL_HEIGHT,relative_to_robot=False)
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
    # robot.go_to_pose(stop_a, relative_to_robot=False).wait_for_completed()
    # robot.go_to_pose(stop_b, relative_to_robot=False).wait_for_completed()
    # robot.go_to_pose(stop_c, relative_to_robot=False).wait_for_completed()
    # robot.go_to_pose(stop_o, relative_to_robot=False).wait_for_completed()

cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)