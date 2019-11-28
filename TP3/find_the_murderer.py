import cozmo
from cozmo.util import degrees,Pose
from inference_class import *
from Map import create_walls, position, create_positions,launch_all


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


def Conclusions():
    # Conclusions
    print("Pièce du crime : ", agent.get_crime_room())
    print("Arme du crime : ", agent.get_crime_weapon())
    print("Personne victime : ", agent.get_victim())
    print("Heure du crime : ", agent.get_crime_hour())
    print("Meurtrier : ", agent.get_suspect())
    print("Personnes innocentes : ", agent.get_innocent())

function_tab = []
function_tab.append(reaction_piece_1)
function_tab.append(reaction_piece_2)
function_tab.append(reaction_piece_3)
function_tab.append(reaction_piece_4)

def analyse_victime():
    # Va vers la victime
    potential_victime = list(set(agent.persons)-set(alive_people))[0]
    print("Est ce que {} est la victime?".format(potential_victime))
    #oui/non
    # if yes :
    agent.add_clause(to_fol(["{} est mort".format(potential_victime)], 'grammars/personne_morte.fcfg'))

    # Quelle est cette pièce?
    str_in = input("Entrez la pièce\n")
    piece = str_in.split(' ')[-1] # quelque soit la phrase, la piece se trouve en dernier
    #print(piece)
    fact = ['{} est dans le {}'.format(agent.get_victim(), piece)]
    agent.add_clause(to_fol(fact, 'grammars/personne_piece.fcfg'))

    # Comment est-elle morte?
    #fact = ['{} est criblée de balles'.format(agent.get_victim())]
    fact = ['{} est atteinte par balle'.format(agent.get_victim())]
    agent.add_clause(to_fol(fact, 'grammars/personne_marque.fcfg'))

    # Demande à Peacock l'heure du decès -> Rep : 14h
    # robot.say_text("A quelle heure est morte {}".format(agent.get_victim()), True, in_parallel=True, duration_scalar=0.5,use_cozmo_voice=True).wait_for_completed() 
    hour = input("Entrez l\'heure\n")
    fact = ['Scarlet est morte à {}h'.format(hour)]
    agent.add_clause(to_fol(fact, 'grammars/personne_morte_heure.fcfg'))

    agent.add_clause('UneHeureApresCrime({})'.format(int(hour)+1))

def suspect_1():
    fact = ['White était dans la cuisine à ' + str(15) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))
    return 0
def suspect_2():
    return 0

def suspect_3():
    # Demande à Mustard dans quelle pièce il était une heure après le meurtre -> Rep : Mustard dans le Garage à 15h
    fact = ['Plum était dans le garage à ' + str(15) + 'h']
    agent.add_clause(to_fol(fact, 'grammars/personne_piece_heure.fcfg'))

    # C'est un fusil que je voit ici?
    #yes/no
    fact = ['Le fusil est dans le garage']
    agent.add_clause(to_fol(fact, 'grammars/arme_piece.fcfg'))


analyse_victime()
suspect_1()
suspect_3()

Conclusions()

############# COZMO PROGRAM
def cozmo_program(robot: cozmo.robot.Robot):
    robot.world.delete_all_custom_objects()
    #create_walls(robot)


    position_tab = create_positions(robot,function_tab)
    launch_all(position_tab)

    Conclusions()

    # for s in stop :
    #     robot.go_to_pose(s, relative_to_robot=False).wait_for_completed()
    #     robot.say_text("Où suis je ?", True, in_parallel=True, duration_scalar=0.5,use_cozmo_voice=True).wait_for_completed() 
    #     piece = input("entrez nom de la pièce")
    #     robot.say_text("je suis à" + piece, True, in_parallel=True, duration_scalar=0.5,use_cozmo_voice=True).wait_for_completed() 



#cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)


# Associer chaque cube_ID à une fonction (victime/buzzer/meurtrier)
# Associer une position à chaque une action
# Pour chaque position
# 
#

