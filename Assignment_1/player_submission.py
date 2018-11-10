#!/usr/bin/env python
from isolation import Board, game_as_text
import random

# This file is your main submission that will be graded against. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.

# Submission Class 1
class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state
        
        Evaluation function that outputs a score equal to how many 
        moves are open for AI player on the board.
            
        Args
            param1 (Board): The board and game state.
            param2 (bool): True if maximizing player is active.

        Returns:
            float: The current state's score. Number of your agent's moves.
            
        """
	
        eval_func = len(game.get_legal_moves())
        return eval_func
        


# Submission Class 2
class CustomEvalFn:

    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state
        
        Custom evaluation function that acts however you think it should. This 
        is not required but highly encouraged if you want to build the best 
        AI possible.
        
        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.
            
        """
        legal_moves = game.get_legal_moves()
        if maximizing_player_turn == True:
            my_moves = len(legal_moves)
            opp_moves = -float("inf")
            for move in legal_moves:
                opp_moves = max(opp_moves, len(game.forecast_move(move).get_legal_moves()))
        else:
            opp_moves = len(legal_moves)
            my_moves = -float("inf")
            for move in legal_moves:
                my_moves = max(my_moves, len(game.forecast_move(move).get_legal_moves()))
        occupied_space = len(game.get_player_locations())
        empty_space = 49-occupied_space
        #return (0.75*my_moves - 0.25*opp_moves)/((empty_space+1))
        return   (my_moves - opp_moves - 0.05*empty_space)/(0.2*occupied_space + 1) 


class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using 
    your evaluation function and 
    a minimax algorithm 
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move."""

    def __init__(self, search_depth=30, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.
        
        if you find yourself with a superior eval function, update the default 
        value of `eval_fn` to `CustomEvalFn()`
        
        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.variable_search_depth = 0
        self.time_to_return = 100
        self.shortter_time_to_return = 70

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent
        
        Args:
            game (Board): The board and game state.
            legal_moves (dict): Dictionary of legal moves and their outcomes
            time_left (function): Used to determine time left before timeout
            
        Returns:
            (tuple): best_move
        """
        #best_move, utility = self.minimax(game, time_left, depth=self.search_depth)
        best_move, utility = self.alphabeta(game, time_left, depth=self.search_depth,alpha=float("-inf"), beta=float("inf"))
        # change minimax to alphabeta after completing alphabeta part of assignment
        return best_move

    def utility(self, game):
        """Can be updated if desired"""
        return self.eval_fn.score(game)
    
    def max_value(self, game, time_left, reached_depth):
        if (not game.get_legal_moves()):
            return -100
        if (time_left() <= self.time_to_return) or (reached_depth <= 0):
            return self.eval_fn.score(game, True)
        v = -float("inf")
        for move in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(move), time_left, reached_depth-1))
        return v
    
    
    
    def min_value(self, game, time_left, reached_depth):
        if (not game.get_legal_moves()):
            return 100
        if (time_left() <= self.time_to_return) or (reached_depth <= 0):
            return self.eval_fn.score(game, False)
        v = float("inf")
        for move in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(move), time_left, reached_depth-1))
        return v


    def minimax(self, game, time_left, depth=3, maximizing_player=True):
        """Implementation of the minimax algorithm
        
        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """
        legal_moves = game.get_legal_moves()
        best_move = None
        best_val = - float("inf")
        previous_best_val = best_val
        level = 1
        while level <= depth:
            if time_left() < self.time_to_return:
                break
            for move in legal_moves:
                if time_left() <= self.time_to_return:
                    break
                best_val = max(best_val, self.min_value(game.forecast_move(move), time_left, level-1))
                if not previous_best_val == best_val:
                    best_move = move
                    previous_best_val = best_val
            level = level + 1
        return best_move, best_val
    
    def max_value_a_b(self, game, time_left, reached_depth, alpha, beta):
        legal_moves = game.get_legal_moves()
        if(not legal_moves):
            return None,-100
        if(time_left() <= self.time_to_return) or (reached_depth <= 0):
            #move = game.get_legal_moves()
            #return move[0], self.eval_fn.score(game.forecast_move(move[0]), True)
            return None, self.eval_fn.score(game, True)
        v = -float("inf")
        previous_alpha = alpha
        best_move = None
        for move in legal_moves:
            v = max(v, self.min_value_a_b(game.forecast_move(move), time_left, reached_depth-1, alpha, beta)[1])
            alpha = max(alpha, v)
            if not (previous_alpha == alpha):
                previous_alpha = alpha
                if (reached_depth == self.variable_search_depth) and (not (move == None)):
                    best_move = move
            if v >= beta:
                return best_move, v
        return best_move, v

    
    def min_value_a_b(self, game, time_left, reached_depth, alpha, beta):
        legal_moves = game.get_legal_moves()
        if(not legal_moves):
            return None, 100
        if(time_left() <= self.time_to_return) or (reached_depth <= 0):
            #move = game.get_legal_moves()
            #return move[0], self.eval_fn.score(game.forecast_move(move[0]), False)
            return None, self.eval_fn.score(game, False)
        v = float("inf")
        previous_beta = beta
        best_move = None
        for move in legal_moves:
            v = min(v, self.max_value_a_b(game.forecast_move(move), time_left, reached_depth-1, alpha, beta)[1])
            beta = min(beta, v)
            if not (previous_beta == beta):
                previous_beta = beta
                if (reached_depth == self.variable_search_depth) and (not (move == None)):
                    best_move = move
            if v <= alpha:
                return best_move, v
        return best_move, v

#    def max_value_a_b(self, game, time_left, reached_depth, alpha, beta):
#        if(not game.get_legal_moves()):
#            return None,-5
#        if(time_left() <= self.time_to_return) or (reached_depth <= 0):
#            return None, self.eval_fn.score(game)
#        v = -float("inf")
#        previous_alpha = alpha
#        best_move = None
#        for move in game.get_legal_moves():
#            v = max(v, self.min_value_a_b(game.forecast_move(move), time_left, reached_depth-1, alpha, beta)[1])
#            alpha = max(alpha, v)
#            if not previous_alpha == alpha:
#                previous_alpha = alpha
#                if reached_depth == self.search_depth and (not move == None):
#                    best_move = move
#            if v >= beta:
#                return best_move, v
#            
#        return best_move, v
#    
#    def min_value_a_b(self, game, time_left, reached_depth, alpha, beta):
#        if(not game.get_legal_moves()):
#            return None, 5
#        if(time_left() <= self.time_to_return) or (reached_depth <= 0):
#            return None, self.eval_fn.score(game)
#        v = float("inf")
#        previous_beta = beta
#        best_move = None
#        for move in game.get_legal_moves():
#            v = min(v, self.max_value_a_b(game.forecast_move(move), time_left, reached_depth-1, alpha, beta)[1])
#            beta = min(beta, v)
#            if not previous_beta == beta:
#                previous_beta = beta
#                if reached_depth == self.search_depth and (not move == None):
#                    best_move = move
#            if v <= alpha:
#                return best_move, v
#        return best_move, v


    def alphabeta(self, game, time_left, depth=3, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implementation of the alphabeta algorithm
        
        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """
        legal_moves = game.get_legal_moves()

        if len(game.get_player_locations()) == 1 or len(game.get_player_locations()) == 3:
            if (3,3) in legal_moves:
                print "yes"
                return (3,3), 1            
            else:
                print "no"
                return legal_moves[random.randint(0,len(legal_moves)-1)], 1        
        best_move = None
        val = -float("inf")
        best_val = val
        previous_best_val = -float("inf")
        level = 3
        while level <= depth:
            self.variable_search_depth = level
            if time_left() <= self.shortter_time_to_return:
                return best_move, best_val
            move, val = self.max_value_a_b(game.copy(), time_left, level, alpha, beta)
            best_val = max(best_val, val)
            if (not (best_val == previous_best_val)) and (not (move == None)):
                best_move = move
                previous_best_val = best_val
            level = level + 1
        return best_move, best_val

#        best_move, best_val = self.max_value_a_b(game.copy(), time_left, depth, alpha, beta)
#        return best_move, best_val



#    def pre_alphabeta(self, game, time_left, depth=3, alpha=float("-inf"), beta=float("inf"),
#                  maximizing_player=True):
#        """Implementation of the alphabeta algorithm
#        
#        Args:
#            game (Board): A board and game state.
#            time_left (function): Used to determine time left before timeout
#            depth: Used to track how deep you are in the search tree
#            alpha (float): Alpha value for pruning
#            beta (float): Beta value for pruning
#            maximizing_player (bool): True if maximizing player is active.
#
#        Returns:
#            (tuple, int): best_move, best_val
#        """
#        legal_moves = game.get_legal_moves()
#        if (3,3) in legal_moves:
#            return (3,3), 1
#        
#        if(not legal_moves):
#            if maximizing_player == True:
#                return None, -100
#            else:
#                return None, 100
#        if(time_left() <= self.time_to_return) or (depth <= 0):
#            return None, self.eval_fn.score(game)
#        if maximizing_player == True:
#            v = -float("inf")
#            previous_alpha = alpha
#            best_move = None
#            for move in legal_moves:
#                v = max(v, self.pre_alphabeta(game.forecast_move(move), time_left, depth-1, alpha, beta, maximizing_player=False)[1])
#                alpha = max(alpha, v)
#                if not(previous_alpha == alpha):
#                    previous_alpha = alpha
#                    if(depth == self.variable_search_depth) and (not(move == None)):
#                        best_move = move
#                if v >= beta:
#                    return best_move, v
#            return best_move, v
#        else:
#            v = float("inf")
#            previous_beta = beta
#            best_move = None
#            for move in legal_moves:
#                v = min(v, self.pre_alphabeta(game.forecast_move(move), time_left, depth-1, alpha, beta, maximizing_player=True)[1])
#                beta = min(beta, v)
#                if not(previous_beta == beta):
#                    previous_beta = beta
#                    if(depth == self.variable_search_depth) and (not(move == None)):
#                        best_move = move
#                if v <= alpha:
#                    return best_move, v
#            return best_move, v
#        
#    def alphabeta(self, game, time_left, depth=3, alpha=float("-inf"), beta=float("inf"),
#                  maximizing_player=True):
#        best_move = None
#        val = -float("inf")
#        best_val = val
#        previous_best_val = -float("inf")
#        level = 4
#        while level <= depth:
#            self.variable_search_depth = level
#            if time_left() <= self.shortter_time_to_return:
#                return best_move, best_val
#            move, val = self.pre_alphabeta(game, time_left, level, alpha, beta)
#            best_val = max(best_val, val)
#            if (not(previous_best_val == best_val)) and (not(move == None)):
#                best_move = move
#                previous_best_val = best_val
#            level = level + 1
#        return best_move, best_val
            