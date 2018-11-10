#!/usr/bin/env python
import traceback
from player_submission import OpenMoveEvalFn, CustomEvalFn, CustomPlayer
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer


#########################################################################################################
#########################################################################################################
class CustomPlayer_1:
    # TODO: finish this class!
    """Player that chooses a move using 
    your evaluation function and 
    a minimax algorithm 
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move."""

    def __init__(self, search_depth=25, eval_fn=OpenMoveEvalFn()):
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
            if not previous_alpha == alpha:
                previous_alpha = alpha
                if (reached_depth == self.variable_search_depth) and (not move == None):
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
            if not previous_beta == beta:
                previous_beta = beta
                if (reached_depth == self.variable_search_depth) and (not move == None):
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
        
        best_move = None
        val = -float("inf")
        best_val = val
        previous_best_val = -float("inf")
        level = 4
        while level <= depth:
            self.variable_search_depth = level
            if time_left() <= self.shortter_time_to_return:
                break
            move, val = self.max_value_a_b(game.copy(), time_left, level, alpha, beta)
            best_val = max(best_val, val)
            if (not best_val == previous_best_val) and (not move == None):
                best_move = move
                previous_best_val = best_val
            level = level + 1
        return best_move, best_val

#        best_move, best_val = self.max_value_a_b(game.copy(), time_left, depth, alpha, beta)
#        return best_move, best_val

    


def main():


    try:
        sample_board = Board(RandomPlayer(), RandomPlayer())
        # setting up the board as though we've been playing
        sample_board.move_count = 1
        sample_board.__board_state__ = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 'Q', 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
        sample_board.__last_queen_move__ = (3,3)
        test = sample_board.get_legal_moves()
        #h = OpenMoveEvalFn()
        h = CustomEvalFn()
        print 'OpenMoveEvalFn Test: This board has a score of %s.' % (h.score(sample_board))
    except NotImplementedError:
        print 'OpenMoveEvalFn Test: Not implemented'
    except:
        print 'OpenMoveEvalFn Test: ERROR OCCURRED'
        print traceback.format_exc()


    try:
        """Example test to make sure
        your minimax works, using the
        #computer_player_moves."""
        # create dummy 5x5 board

        p1 = CustomPlayer()
        p2 = CustomPlayer(search_depth=3)
        #p2 = HumanPlayer()
        b = Board(p1, p2, 5, 5)
        b.__board_state__ = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 'Q', 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        b.__last_queen_move__ = (2, 2)
       
        b.move_count = 1

        output_b = b.copy()
        winner, move_history, termination = b.play_isolation_name_changed()
        print 'Minimax Test: Runs Successfully'
	print winner
        # Uncomment to see example game
        #print game_as_text(winner, move_history,  termination, output_b)
    except NotImplementedError:
        print 'Minimax Test: Not Implemented'
    except:
        print 'Minimax Test: ERROR OCCURRED'
        print traceback.format_exc()



    """Example test you can run
    to make sure your AI does better
    than random."""
    try:
        r = CustomPlayer_1(8)
#        h = RandomPlayer()
        
        h = CustomPlayer()
        #r = RandomPlayer()
        game = Board(r, h, 7, 7)
        output_b = game.copy()
        winner, move_history, termination = game.play_isolation_name_changed()
        if 'CustomPlayer' in str(winner):
            print 'CustomPlayer Test: CustomPlayer Won'
        else:
            print 'CustomPlayer Test: CustomPlayer Lost'
        # Uncomment to see game
        print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'CustomPlayer Test: Not Implemented'
    except:
        print 'CustomPlayer Test: ERROR OCCURRED'
        print traceback.format_exc()

if __name__ == "__main__":
    main()
