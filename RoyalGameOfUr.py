#royal game of Ur, Finkel Rules
#https://royalur.net/rules


import random

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
p1_score = 0
p2_score = 0


#passes the correct argument into the next turn
def calibrate_turn_variables(current_player):

    print(f"\ncalibrating turn parameters for player {current_player}\n")
    
    if current_player == 1:
        turn(starting_zonep1, home_zonep1, p1_pieces, p2_pieces, p1_score, dark, 2, 1)
    else:
        turn(starting_zonep2, home_zonep2, p2_pieces, p1_pieces, p2_score, light, 1, 2)

             
#all possible moves and steps for a turn
def turn(starting_zone, home_zone, player_pieces, opponent_pieces, score, color, next_player, current_player):

    print(f"starting turn for player {current_player}")

    #1) roll die: random choice between 0 and 4
    roll = random.randint(0, 4)
    print(f"Roll: {roll}")

    #if no pieces on board for current player, place one
    if 0 not in player_pieces and roll != 0:
        print("No pieces on board. Placing one.")
        starting_zone[roll-1] = color
        player_pieces[0] = 0
        print(f"Player {current_player} pieces: {player_pieces}")
        if  (roll) == 4: #if landed on rosette, go again
            print(f"Landed on rosette. Player {current_player} goes again.")
            turn(starting_zone, home_zone, player_pieces, opponent_pieces, score, color, next_player, current_player)
        else: #else, on to next player's turn
            post_move(current_player, next_player, score)
    
    # for all pieces that can move, ask for user input on which to move
    # write moves first, then designate them to their own fucntions that the user can choose from.
    # if opponent is on rosetta, they're safe 
    #determine legal possible moves
    #list of start space, start desc, end space, end desc, capture (T/F), land on rosette (T/F)
    possible_moves = []

    print("determining possible moves")

    ##### movement & capture of pieces already on board #####
    #indices for pieces in starting zone (if any)
    #list of positions
    if roll != 0:
        sz_pieces = [(idx + 1) for idx, space in enumerate(starting_zone) if space == color]
        for idx in sz_pieces: 
            #if movement would go into combat zone
            if (idx + roll) > len(starting_zone):
                #if the player does not have a piece there
                if combat_zone[idx + roll - len(starting_zone) -1] != color:
                    #if enemy has a piece there
                    if combat_zone[idx + roll - len(starting_zone) -1] != 0:
                        possible_moves.append([starting_zone[idx - 1], f"Starting zone space {idx}", combat_zone[idx + roll - len(starting_zone) -1], f"Combat zone space {idx + roll - len(starting_zone)}", True, False])
                    #if enemy does not have piece there
                    else: 
                        #if rosette
                        if (idx + roll - len(starting_zone) -1) == 3: 
                            possible_moves.append([starting_zone[idx - 1], f"Starting zone space {idx}", combat_zone[idx + roll - len(starting_zone) -1], f"Combat zone space {idx + roll - len(starting_zone)}", False, True])
                        #if not rosette
                        else: 
                            possible_moves.append([starting_zone[idx - 1], f"Starting zone space {idx}", combat_zone[idx + roll - len(starting_zone) -1], f"Combat zone space {idx + roll - len(starting_zone)}", False, False])
            #if movement would stay in starting zone
            else:
                #if not a piece there already
                if starting_zone[idx + roll] == 0:
                    #if space is rosette: 
                    if (idx + roll) == 4:
                        possible_moves.append([starting_zone[idx - 1], f"Starting zone space {idx}", starting_zone[idx + roll - 1], f"Starting zone space {idx + roll}", False, True])
                    #if space is not rosette
                    else:
                        possible_moves.append([starting_zone[idx - 1],f"Starting zone space {idx}", starting_zone[idx + roll - 1],f"Starting zone space {idx + roll}", False, False])
    

    #####introduce a new piece#####

    #####list moves #####
    print("displaying possible moves...")
    if len(possible_moves) == 0: 
        print("No possible moves; skip this turn.")
    else: 
        print("Possible moves: ")
        for move in possible_moves: 
            print(f"{move[1]} to {move[3]}")


    #####scoring conditions#####
    #if in combat_zone
    if combat_zone[-1] == color and roll == 4:
        score += 1
        combat_zone[-1] = 0
    #if in home_zone
    for space in home_zone:
        if space == color and home_zone[space + (roll-1)] == 10: 
            score += 1
            home_zone[space] = 0

def post_move(current_player, next_player, score):
    #win-check
    if score == 7:
        print(f"Player {current_player} has won! 🎉")

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
            
    print("board:\n")
    for row in board_vis:
        print(" ".join(row))

    #finally, change current player
    current_player = next_player
    print(f"Turn concluded. Next player is {current_player}")
    calibrate_turn_variables(current_player)
   

#initialize game: choose who rolls first
current_player = random.randint(1, 2)
print(f"Player {current_player} goes first")
calibrate_turn_variables(current_player)


