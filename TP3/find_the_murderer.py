import cozmo
import time
from cozmo.util import degrees,Pose
from inference_class import *
from Map import create_walls, position, create_positions,launch_all

cube_taps = 0

def handle_object_tapped(evt, **kw):
    global cube_taps 
    #print(evt.obj.object_id)
    # This will be called whenever an EvtObjectMovingStarted event is dispatched -
    # whenever we detect a cube starts moving (via an accelerometer in the cube)
    if evt.obj.object_id ==2 :
        cube_taps = cube_taps + evt.tap_count if cube_taps<3 else 0
        print(cube_taps)

# Parcours fixe mais objet diff dans chaque 
# trouve un mqrqueyr => quel est l'objet 
# trouve le cube victime => demande qui c'est 

agent = CrimeInference()

alive_people = ['Mustard','Peacock', 'Plum', 'White']

for a_p in alive_people:    
    agent.add_clause(to_fol(["{} est vivant".format(a_p)], 'grammars/personne_vivant.fcfg'))


def reaction_piece_1(robot: cozmo.robot.Robot):
    # On se rend compte que Scarlet est morte par étranglement
    fact = ['Scarlet a des marques au cou']
    agent.add_clause(to_fol(fact, 'grammars/personne_marque.fcfg'))
    fact = ['Scarlet est dans le bureau']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece.fcfg'))

    # On sait que Peacock est dans le bureau
    fact = ['Peacock est dans le bureau']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece.fcfg'))

    # Demande à Peacock l'heure du decès -> Rep : 14h
    robot.say_text("A quelle heure est morte {}".format(agent.get_victim()), True, in_parallel=True, duration_scalar=0.5,use_cozmo_voice=True).wait_for_completed() 
    fact = input("Entrez l\'heure")
    #fact = ['Scarlet est morte à 14h']
    agent.add_clause(to_fol(fact, 'grammars/personne_morte_heure.fcfg'))

    uneHeureApres = agent.get_crime_hour() + 1

    agent.add_clause('UneHeureApresCrime({})'.format(uneHeureApres))

    # Demande à Peacock dans quelle pièce il était une heure après le meurtre -> Rep : Peacock dans le Salon à 15h
    fact = ['Peacock était dans le salon à ' + str(uneHeureApres) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))

def reaction_piece_2(robot: cozmo.robot.Robot):
    # Dans le salon
    # Voit qu'il y a un fusil et Plum dans le salon
    fact = ['Le fusil est dans le salon']
    agent.add_clause(to_fol(fact, 'grammars/arme_piece.fcfg'))
    fact = ['Plum est dans le salon']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece.fcfg'))

    uneHeureApres = agent.get_crime_hour() + 1

    # Demande à Plum dans quelle pièce il était une heure après le meurtre -> Rep : Plum dans le Salon à 15h
    fact = ['Plum était dans le salon à ' + str(uneHeureApres) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))

def reaction_piece_3(robot: cozmo.robot.Robot):
    # Dans la cuisine
    # Voit qu'il y a un couteau, White et Mustard dans la cuisine
    fact = ['Le couteau est dans la cuisine']
    agent.add_clause(to_fol(fact, 'grammars/arme_piece.fcfg'))
    fact = ['White est dans la cuisine']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece.fcfg'))
    fact = ['Mustard est dans la cuisine']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece.fcfg'))

    uneHeureApres = agent.get_crime_hour() + 1

    # Demande à White dans quelle pièce il était une heure après le meurtre -> Rep : White dans la Cuisine à 15h
    fact = ['White était dans la cuisine à ' + str(uneHeureApres) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))

    # Demande à Mustard dans quelle pièce il était une heure après le meurtre -> Rep : Mustard dans le Garage à 15h
    fact = ['Mustard était dans le garage à ' + str(uneHeureApres) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))

def reaction_piece_4(robot: cozmo.robot.Robot):
    # Dans le garage
    # On se rend compte qu'il y a une corde dans le garage
    fact = ['La corde est dans le garage']
    agent.add_clause(to_fol(fact, 'grammars/arme_piece.fcfg'))



function_tab = []
function_tab.append(reaction_piece_1)
function_tab.append(reaction_piece_2)
function_tab.append(reaction_piece_3)
function_tab.append(reaction_piece_4)


def analyse_victime(robot):
    global cube_taps
    # Va vers la victime
    potential_victime = list(set(agent.persons)-set(alive_people))[0]
    robot.say_text("Est ce que {} est la victime?".format(potential_victime)).wait_for_completed()
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

def suspect_1(robot):
    hour_plus_one = agent.get_crime_hour_plus_one()

    robot.say_text("Qui êtes vous?").wait_for_completed()
    str_in = input("Entrez son nom\n")
    name = str_in.split(' ')[-1] # quelque soit la phrase, le nom se trouve en dernier

    fact = ['{} était dans la cuisine à '.format(name) + str(hour_plus_one) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))

    # Voit un couteau
    fact = ['Le couteau est dans la cuisine']
    robot.say_text(fact[0]).wait_for_completed()
    agent.add_clause(to_fol(fact, 'grammars/arme_piece.fcfg'))

    return 0
def suspect_2(robot):
    hour_plus_one = agent.get_crime_hour_plus_one()

    robot.say_text("Je vois une corde dans le garage").wait_for_completed()
    fact = ['Le corde est dans la garage']
    agent.add_clause(to_fol(fact, 'grammars/arme_piece.fcfg'))

    robot.say_text("Qui êtes vous?").wait_for_completed()
    str_in = input("Entrez son nom\n")
    name = str_in.split(' ')[-1] # quelque soit la phrase, le nom se trouve en dernier

    fact = ['{} était dans le garage à '.format(name) + str(hour_plus_one) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))

    return 0

def suspect_3(robot):
    global cube_taps
    hour_plus_one = agent.get_crime_hour_plus_one()
    potential_arme = agent.get_crime_weapon()
    # Demande à Mustard dans quelle pièce il était une heure après le meurtre -> Rep : Mustard dans le Garage à 15h
    robot.say_text("Qui êtes vous?").wait_for_completed()
    str_in = input("Entrez son nom\n")
    name = str_in.split(' ')[-1] # quelque soit la phrase, le nom se trouve en dernier

    robot.say_text("Où étiez vous à {} heure?".format(hour_plus_one)).wait_for_completed()
    str_in = input("Entrez la pièce\n")
    piece = str_in.split(' ')[-1] # quelque soit la phrase, la piece se trouve en dernier


    fact = ['{} était dans le {} à '.format(name,piece) + str(hour_plus_one) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))

    # C'est un fusil que je voit ici?
    robot.say_text("C'est votre {} que je vois?".format(potential_arme)).wait_for_completed()
    print("En attente d'une réponse oui/non\n")
    time.sleep(3)
    if cube_taps == 1:
        agent.add_clause(to_fol(["Le {} est dans le {}".format(potential_arme,piece)], 'grammars/arme_piece.fcfg'))

    cube_taps=0
    

def Conclusions():
    # Conclusions
    print("Pièce du crime : ", agent.get_crime_room())
    print("Arme du crime : ", agent.get_crime_weapon())
    print("Personne victime : ", agent.get_victim())
    print("Heure du crime : ", agent.get_crime_hour())
    print("Meurtrier : ", agent.get_suspect())
    print("Personnes innocentes : ", agent.get_innocent())

############# COZMO PROGRAM
def cozmo_program(robot: cozmo.robot.Robot):
    robot.world.delete_all_custom_objects()
    robot.add_event_handler(cozmo.objects.EvtObjectTapped, handle_object_tapped)

    analyse_victime(robot)
    suspect_1(robot)
    suspect_2(robot)
    suspect_3(robot)
    Conclusions()



    #create_walls(robot)


    # position_tab = create_positions(robot,function_tab)
    # launch_all(position_tab)

    # Conclusions()

    # for s in stop :
    #     robot.go_to_pose(s, relative_to_robot=False).wait_for_completed()
    #     robot.say_text("Où suis je ?", True, in_parallel=True, duration_scalar=0.5,use_cozmo_voice=True).wait_for_completed() 
    #     piece = input("entrez nom de la pièce")
    #     robot.say_text("je suis à" + piece, True, in_parallel=True, duration_scalar=0.5,use_cozmo_voice=True).wait_for_completed() 



# cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)


# Associer chaque cube_ID à une fonction (victime/buzzer/meurtrier)
# Associer une position à chaque une action
# Pour chaque position
# 
#

