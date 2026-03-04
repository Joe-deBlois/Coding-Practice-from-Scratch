#royal game of Ur, Finkel Rules
#https://royalur.net/rules


import random
import sys

#define zones of the board
starting_zonep1 = [0, 0, 0, 0] #idx 3 is rosette, space 4
starting_zonep2 = [0, 0, 0, 0] #idx 3 is rosette, space 4
combat_zone = [0, 0, 0, 0, 0, 0, 0, 0] #idx 3 is rosette, space 4
home_zonep1 = [0, 0, 10]    #idx 1 is rosette (space 2), idx 2 scores (value 10)
home_zonep2 = [0, 0, 10]    #idx 1 is rosette (space 2), idx 2 scores (value 10)

#pieces & score
dark = "dark"
light = "light"

p1_pieces = [dark, dark, dark, dark, dark, dark, dark]
p2_pieces = [light, light, light, light, light, light, light]
score_vals = {1: 0, 2: 0}

#passes the correct argument into the next turn
def calibrate_turn_variables(current_player):

    #print(f"\ncalibrating turn parameters for player {current_player}\n")
    
    if current_player == 1:
        return turn(starting_zonep1, home_zonep1, p1_pieces, p2_pieces, score_vals, dark, light, 2, 1)
    else:
        return turn(starting_zonep2, home_zonep2, p2_pieces, p1_pieces, score_vals, light, dark, 1, 2)

             
#all possible moves and steps for a turn
def turn(starting_zone, home_zone, player_pieces, opponent_pieces, score_vals, color, opp_color, next_player, current_player):

    print(f"It is {current_player}'s turn\n__________________________________________")

    #1) roll die: random choice between 0 and 4
    roll = random.choices(
        population = [0, 1, 2, 3, 4], 
        k = 1, 
        weights = [(1/16), (4/16), (6/16), (4/16), (1/16)] )[0]
    print(f"Roll: {roll}")

    #if no pieces on board for current player, place one
    if 0 not in player_pieces and roll != 0:
        input("No pieces on board. Placing one.\tClick enter.")
        starting_zone[roll-1] = color
        player_pieces[0] = 0
        if  (roll) == 4: #if landed on rosette, go again
            input(f"Landed on rosette. Player {current_player} goes again.\tClick enter.")
            return post_move(current_player, current_player, score_vals)
        else: #else, on to next player's turn
            return post_move(current_player, next_player, score_vals)
    
    # for all pieces that can move, ask for user input on which to move
    # write moves first, then designate them to their own fucntions that the user can choose from.
    # if opponent is on rosetta, they're safe 
    #determine legal possible moves
    #list of dictionaries.
    #keys are "from_zone", "from_idx", "to_zone", "to_idx", "capture", "rosette", "score", "from_desc", "to_desc"
    possible_moves = []

    ##### movement & capture of pieces already on board #####
   
    if roll != 0:
        ###starting zone###
        sz_pieces = [(idx + 1) for idx, space in enumerate(starting_zone) if space == color]
        for space in sz_pieces: 
            #if movement would go into combat zone
            if (space + roll) > len(starting_zone):
                #if the player does not have a piece there
                if combat_zone[space + roll - len(starting_zone) -1] != color:
                    #if enemy has a piece there
                    if combat_zone[space + roll - len(starting_zone) -1] != 0 and (space + roll - len(home_zone)) != 4:
                        possible_moves.append(
                            {"from_zone" : "starting",
                            "from_idx" : space - 1, 
                            "to_zone" : "combat", 
                            "to_idx" : space + roll - len(starting_zone) -1, 
                            "capture": True, 
                            "rosette": False, 
                            "from_desc": f"starting zone space {space}", 
                            "to_desc": f"combat zone space {space + roll - len(starting_zone)}"})
                    #if enemy does not have piece there
                    else: 
                        #if rosette
                        if (space + roll - len(starting_zone)) == 4: 
                            possible_moves.append(
                             {"from_zone" : "starting",
                            "from_idx" : space - 1, 
                            "to_zone" : "combat", 
                            "to_idx" : space + roll - len(starting_zone) -1, 
                            "capture": False, 
                            "rosette": True, 
                            "from_desc": f"starting zone space {space}", 
                            "to_desc": f"combat zone space {space + roll - len(starting_zone)}"})
                        #if not rosette
                        else: 
                            possible_moves.append(
                            {"from_zone" : "starting",
                            "from_idx" : space - 1, 
                            "to_zone" : "combat", 
                            "to_idx" : space + roll - len(starting_zone) -1, 
                            "capture": False, 
                            "rosette": False, 
                            "from_desc": f"starting zone space {space}", 
                            "to_desc": f"combat zone space {space + roll - len(starting_zone)}"})
            #if movement would stay in starting zone
            else:
                #if not a piece there already
                if starting_zone[space + roll - 1] == 0:
                    #if space is rosette: 
                    if (space + roll) == 4:
                        possible_moves.append(
                            {"from_zone" : "starting",
                            "from_idx" : space - 1, 
                            "to_zone" : "starting", 
                            "to_idx" : space + roll - 1, 
                            "capture": False, 
                            "rosette": True, 
                            "from_desc": f"starting zone space {space}", 
                            "to_desc": f"starting zone space {space + roll}"})
                    #if space is not rosette
                    else:
                        possible_moves.append(
                            {"from_zone" : "starting",
                            "from_idx" : space - 1, 
                            "to_zone" : "starting", 
                            "to_idx" : space + roll - 1, 
                            "capture": False, 
                            "rosette": False, 
                            "from_desc": f"starting zone space {space}", 
                            "to_desc": f"starting zone space {space + roll}"})
        
        ###combat zone###
        cz_pieces = [(idx + 1) for idx, space in enumerate(combat_zone) if space == color]
        for space in cz_pieces: 
            #if movement would go outside combat zone
            if ((space + roll) > len(combat_zone)):
                #if movement would fall in home zone
                if (space + roll - len(combat_zone) <= len(home_zone)):
                    #if the player does not have a piece there
                    if home_zone[space + roll - len(combat_zone) -1] != color:
                        #if score
                        if home_zone[space + roll - len(combat_zone) -1] == 10: 
                                possible_moves.append(
                            {"from_zone" : "combat",
                            "from_idx" : space - 1, 
                            "to_zone" : "score", 
                            "to_idx" : "score", 
                            "capture": False, 
                            "rosette": False, 
                            "from_desc": f"combat zone space {space}", 
                            "to_desc": "score"})
                        else: 
                            if space + roll - len(combat_zone) == 1:
                                possible_moves.append(
                                {"from_zone" : "combat",
                                "from_idx" : space - 1, 
                                "to_zone" : "home", 
                                "to_idx" : space + roll - len(combat_zone)-1, 
                                "capture": False, 
                                "rosette": False, 
                                "from_desc": f"combat zone space {space}", 
                                "to_desc": f"home zone space {space + roll - len(combat_zone)}"})
                            if space + roll - len(combat_zone) == 2: 
                                possible_moves.append(
                                {"from_zone" : "combat",
                                "from_idx" : space - 1, 
                                "to_zone" : "home", 
                                "to_idx" : space + roll - len(combat_zone)-1, 
                                "capture": False, 
                                "rosette": True, 
                                "from_desc": f"combat zone space {space}", 
                                "to_desc": f"home zone space {space + roll - len(combat_zone)}"})

            #if movement would stay in combat zone
            else:
                #if not a piece there already
                if combat_zone[space + roll -1] == 0:
                    #if space is rosette: 
                    if (space + roll) == 4:
                        possible_moves.append(
                            {"from_zone" : "combat",
                            "from_idx" : space - 1, 
                            "to_zone" : "combat", 
                            "to_idx" : space + roll - 1, 
                            "capture": False, 
                            "rosette": True, 
                            "from_desc": f"combat zone space {space}", 
                            "to_desc": f"combat zone space {space + roll}"})
                    #if space is not rosette
                    else:
                        possible_moves.append(
                            {"from_zone" : "combat",
                            "from_idx" : space - 1, 
                            "to_zone" : "combat", 
                            "to_idx" : space + roll - 1, 
                            "capture": False, 
                            "rosette": False, 
                            "from_desc": f"combat zone space {space}", 
                            "to_desc": f"combat zone space {space + roll}"})
                #if opponent piece is there & can be taken
                if combat_zone[space + roll -1] == opp_color and (space + roll) != 4:
                    possible_moves.append(
                            {"from_zone" : "combat",
                            "from_idx" : space - 1, 
                            "to_zone" : "combat", 
                            "to_idx" : space + roll - 1, 
                            "capture": True, 
                            "rosette": False, 
                            "from_desc": f"combat zone space {space}", 
                            "to_desc": f"combat zone space {space + roll}"})
    
        ###home zone###
        hz_pieces = [(idx + 1) for idx, space in enumerate(home_zone) if space == color]
        for space in hz_pieces: 
            if (roll + space) <= len(home_zone):
                #if piece can score
                if home_zone[space + roll -1 ] == 10:
                    possible_moves.append(
                            {"from_zone" : "home",
                            "from_idx" : space - 1, 
                            "to_zone" : "score", 
                            "to_idx" : "score", 
                            "capture": False, 
                            "rosette": False, 
                            "from_desc": f"home zone space {space}", 
                            "to_desc": "score"})
                else: 
                    #if already in home & not scoring, only other open space must be rosette
                    if home_zone[space + roll - 1] == 0: 
                        possible_moves.append(
                            {"from_zone" : "home",
                            "from_idx" : space - 1, 
                            "to_zone" : "home", 
                            "to_idx" : space + roll - 1, 
                            "capture": False, 
                            "rosette": True, 
                            "from_desc": f"home zone space {space}", 
                            "to_desc": f"home zone space {space + roll}"})

        #####introduce a new piece#####
        #if player still has pieces to introduce
        num_pieces = 0
        for p in player_pieces: 
            if p == color:
                num_pieces +=1
        if num_pieces > 0: 
            if starting_zone[roll - 1] == 0: 
                #if space is free
                if starting_zone[roll - 1] == 0: 
                    #if rosette
                    if roll == 4: 
                        possible_moves.append(
                            {"from_zone" : "place piece",
                            "from_idx" : "place piece", 
                            "to_zone" : "starting", 
                            "to_idx" : roll - 1, 
                            "capture": False, 
                            "rosette": True, 
                            "from_desc": "place piece", 
                            "to_desc": f"starting zone space {roll}"})

                    else: 
                        possible_moves.append(
                            {"from_zone" : "place piece",
                            "from_idx" : "place piece", 
                            "to_zone" : "starting", 
                            "to_idx" : roll - 1, 
                            "capture": False, 
                            "rosette": False, 
                            "from_desc": "place piece", 
                            "to_desc": f"starting zone space {roll}"})

    #####list moves #####
    counter = 0
    if len(possible_moves) == 0: 
        input("No possible moves; skip this turn.\tClick enter.")
        return post_move(current_player, next_player, score_vals)
    else: 
        print("\nPossible moves: \n- - - - - - - - - - - - - - - - - - - - - -")
        for move in possible_moves: 
            print(f"{counter}\t{move['from_desc']} to {move['to_desc']}")
            counter += 1

    #####player chooses move#####0

    user_input = input("What is your choice? ")
    while user_input not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        user_input = input("What is your choice? ")
    print("\n")
    player_choice = possible_moves[int(user_input)]


    #calibrate variables from player choice#

    from_zone = get_zone(player_choice["from_zone"], starting_zone, home_zone, combat_zone)
    to_zone = get_zone(player_choice["to_zone"], starting_zone, home_zone, combat_zone)

    from_index = player_choice["from_idx"]
    to_index = player_choice["to_idx"]

    #make player move#

    #1. score
    if player_choice["to_idx"] == "score":
        score_vals[current_player] +=1
    #2. move
    if player_choice["to_idx"] != "score":
        to_zone[to_index] = color
        

    #3. "delete" old piece
    if player_choice["from_zone"] == "place piece":
        #select a piece to enter the board
        piece_idx = next((p for p, q in enumerate(player_pieces) if q != 0), None)
        player_pieces[piece_idx] = 0
        
    if player_choice["from_zone"] != "place piece":
        from_zone[from_index] = 0
    


    #4. capture
    if player_choice["capture"] == True:
        combat_zone[to_index] = color
        #choose an opponent's piece slot to be refilled
        captured_index = next((i for i, v in enumerate(opponent_pieces) if v == 0), None)
        if captured_index is not None: 
            opponent_pieces[captured_index] = opp_color

    #4. rosette 
    if player_choice["rosette"] == True: 
        input(f"Landed on rosette. Player {current_player} goes again.\tClick enter.")
        #player goes again
        return post_move(current_player, current_player, score_vals)
    return post_move(current_player, next_player, score_vals)


def get_zone(zone_name, starting_zone, home_zone, combat_zone): 
    if zone_name == "starting":
        return starting_zone
    if zone_name == "combat":
        return combat_zone
    if zone_name == "home":
        return home_zone
    if zone_name == "score": 
        return "score"
    if zone_name == "place piece":
        return "place piece"
  
    
def post_move(current_player, next_player, score_vals):
    #win-check
    if score_vals[current_player] >= 7:
        winning_condition_met(current_player)
    if score_vals[next_player] >= 7:
        winning_condition_met(next_player)

    #visualize the board now
    board = [
        [starting_zonep1[3], combat_zone[0], starting_zonep2[3]], 
        [starting_zonep1[2], combat_zone[1], starting_zonep2[2]], 
        [starting_zonep1[1], combat_zone[2], starting_zonep2[1]], 
        [starting_zonep1[0], combat_zone[3], starting_zonep2[0]], 
                        ["", combat_zone[4], ""], 
                        ["", combat_zone[5], ""], 
        [home_zonep1[1], combat_zone[6], home_zonep2[1]], 
        [home_zonep1[0], combat_zone[7], home_zonep2[0]]]

    board_vis = [[] for _ in range(len(board))]

    for i, row in enumerate(board):
        for space in row:
            if space == 0:
                board_vis[i].append("▒▒")
            elif space == "dark":
                board_vis[i].append("⬛")
            elif space == "light":
                board_vis[i].append("⬜")
            elif space == "":
                board_vis[i].append("  ")

    #add rosettes to empty rosette spaces
    if starting_zonep1[3] == 0:
        board_vis[0][0] = "🟦"
    if starting_zonep2[3] == 0: 
        board_vis[0][2] = "🟦"
    if combat_zone[3] == 0:
        board_vis[3][1] = "🟦"
    if home_zonep1[1] == 0:
        board_vis[6][0] = "🟦"
    if home_zonep2[1] == 0:
        board_vis[6][2] = "🟦"
            
    print("\nboard:\n")
    for row in board_vis:
        print(" ".join(row))

    global p1_pieces, p2_pieces
    print(f"\n\tp1 score: {score_vals[1]}\tp1 pieces: {p1_pieces.count("dark")}\n\tp2 score: {score_vals[2]}\tp2 pieces: {p2_pieces.count("light")}")

    #finally, change current player
    print(f"Turn concluded. Next player is {next_player}")
    return calibrate_turn_variables(next_player)

def winning_condition_met(winner):
    print(f"- - - - - - - - - - - -- - - -\n\nPlayer {winner} has won! 🎉\n\n- - - - - - - - - - - - - - - -")
    sys.exit()
   

#initialize game: choose who rolls first & begin their move
current_player = random.randint(1, 2)
print(f"Player {current_player} goes first")
calibrate_turn_variables(current_player)
    