import cozmo
import time
from cozmo.util import degrees,Pose
from inference_class import *

cube_taps = 0
#cube1 = robot.world.get_light_cube(LightCube1Id)
def handle_object_tapped(evt, **kw):
    global cube_taps 
    #print(evt.obj.object_id)
    # This will be called whenever an EvtObjectMovingStarted event is dispatched -
    # whenever we detect a cube starts moving (via an accelerometer in the cube)
    if evt.obj.cube_id == cozmo.objects.LightCube1Id:
        cube_taps = cube_taps + evt.tap_count if cube_taps<2 else 0
        print(cube_taps)

agent = CrimeInference()

alive_people = ['Mustard','Peacock', 'Plum', 'White']

for a_p in alive_people:    
    agent.add_clause(to_fol(["{} est vivant".format(a_p)], 'grammars/personne_vivant.fcfg'))

def Filter(string, substr): 
    return [s for s in string if s not in substr]

def analyse_victime(robot):
    global cube_taps
    # Va vers la victime
    potential_victime = Filter(agent.persons,alive_people)[0]
    robot.say_text("Est ce que {} est la victime?".format(potential_victime)).wait_for_completed()
    cube_taps=0
    print("En attente d'une réponse oui/non\n")
    time.sleep(3)
    if cube_taps == 1:
        agent.add_clause(to_fol(["{} est mort".format(potential_victime)], 'grammars/personne_morte.fcfg'))
    cube_taps=0

    # Quelle est cette pièce?
    robot.say_text("Quelle est cette pièce?".format(potential_victime)).wait_for_completed()
    str_in = input("Entrez la pièce\n")
    piece = str_in.split(' ')[-1] # quelque soit la phrase, la piece se trouve en dernier
    
    fact = ['{} est dans le {}'.format(potential_victime, piece)]
    print(fact)
    agent.add_clause(to_fol(fact, 'grammars/personne_piece.fcfg'))
    
    # Comment est-elle morte?
    fact = ['{} est criblée de balles'.format(agent.get_victim())]
    robot.say_text(fact[0]).wait_for_completed()
    agent.add_clause(to_fol(fact, 'grammars/personne_marque.fcfg'))

    # Demande à Peacock l'heure du decès -> Rep : 14h
    robot.say_text("Quelles est l'heure du décés?").wait_for_completed()
    hour = input("Entrez l\'heure\n")
    fact = ['{} est morte à {}h'.format(potential_victime,hour)]
    agent.add_clause(to_fol(fact, 'grammars/personne_morte_heure.fcfg'))

    agent.add_clause('UneHeureApresCrime({})'.format(int(hour)+1))

def indice_1(robot):
    hour_plus_one = agent.get_crime_hour_plus_one()
    if hour_plus_one<12:
        part_of_day = "ce matin"
    elif hour_plus_one <18:
        part_of_day = "cette après midi"
    else :
        part_of_day = "ce soir"

    # Voit un couteau
    fact = ['Le couteau est dans la cuisine']
    robot.say_text("Je vois un couteau dans la cuisine").wait_for_completed()
    agent.add_clause(to_fol(fact, 'grammars/arme_piece.fcfg'))

    robot.say_text("Qui s'est occupé de la cuisine {}?".format(part_of_day)).wait_for_completed()
    str_in = input("Entrez son nom\n")
    name = str_in.split(' ')[-1] # quelque soit la phrase, le nom se trouve en dernier

    fact = ['{} était dans la cuisine à '.format(name) + str(hour_plus_one) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))
    fact = ['{} était dans la cuisine'.format(name)]
    agent.add_clause(to_fol(fact, 'grammars/personne_piece.fcfg'))
  
    return 0

def indice_2(robot):
    hour_plus_one = agent.get_crime_hour_plus_one()
    if hour_plus_one<12:
        part_of_day = "ce matin"
    elif hour_plus_one <18:
        part_of_day = "cette après midi"
    else :
        part_of_day = "ce soir"

    robot.say_text("Je vois une corde dans le garage").wait_for_completed()
    fact = ['Le corde est dans la garage']
    agent.add_clause(to_fol(fact, 'grammars/arme_piece.fcfg'))

    robot.say_text("Qui est allé dans le garage {}?".format(part_of_day)).wait_for_completed()
    str_in = input("Entrez son nom\n")
    name = str_in.split(' ')[-1] # quelque soit la phrase, le nom se trouve en dernier

    fact = ['{} était dans le garage à '.format(name) + str(hour_plus_one) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))
    fact = ['{} était dans le garage'.format(name) ]
    agent.add_clause(to_fol(fact, 'grammars/personne_piece.fcfg'))

    return 0

def indice_3(robot):
    global cube_taps
    hour_plus_one = agent.get_crime_hour_plus_one()
    potential_arme = agent.get_crime_weapon()


    suspects = Filter(agent.persons, agent.get_innocent())
    # C'est un fusil que je voit ici?
    robot.say_text("Je vois l'arme du crime.".format(potential_arme)).wait_for_completed()

    cube_taps=0
    robot.say_text("Ce doite être le fusil de {} ?".format(suspects[0])).wait_for_completed()
    print("En attente d'une réponse oui/non\n")
    time.sleep(3)
    if cube_taps == 1:
        # agent.add_clause(to_fol(["Le {} est dans le {}".format(potential_arme,piece)], 'grammars/arme_piece.fcfg'))
        suspect = suspects[0]
    else :
        robot.say_text("Alors c'est celui de {} ?".format(suspects[1])).wait_for_completed()
        suspect = suspects[1]
    cube_taps=0

    robot.say_text("Où été {} à {} heure?".format(suspect, hour_plus_one)).wait_for_completed()
    str_in = input("Entrez la pièce\n")
    piece = str_in.split(' ')[-1] # quelque soit la phrase, la piece se trouve en dernier
    agent.add_clause(to_fol(["Le {} est dans le {}".format(potential_arme,piece)], 'grammars/arme_piece.fcfg'))

    fact = ['{} était dans le {} à '.format(suspect,piece) + str(hour_plus_one) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))

def Conclusions():
    # Conclusions
    print("Pièce du crime : ", agent.get_crime_room())
    print("Arme du crime : ", agent.get_crime_weapon())
    print("Personne victime : ", agent.get_victim())
    print("Heure du crime : ", agent.get_crime_hour())
    print("Meurtrier : ", agent.get_suspect())
    print("Personnes innocentes : ", agent.get_innocent()) 


############# COZMO PROGRAM FOR test
def cozmo_program(robot: cozmo.robot.Robot):
    robot.world.delete_all_custom_objects()
    robot.add_event_handler(cozmo.objects.EvtObjectTapped, handle_object_tapped)

    #create_walls(robot)

    analyse_victime(robot)
    indice_1(robot)
    indice_2(robot)
    indice_3(robot)
    Conclusions()

cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)


