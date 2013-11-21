import random

from org.ggp.base.util.statemachine import MachineState
from org.ggp.base.util.statemachine.implementation.prover import ProverStateMachine
from org.ggp.base.player.gamer.statemachine import StateMachineGamer

class PythonHeuristicGamer(StateMachineGamer):


    def preview(self, game, timeout):
        print "gettin some preview!"
    def getName(self):
        pass
        
    def stateMachineMetaGame(self, timeout):
        pass


    def minmax_move(self, role, state,
            alpha = float("-inf"), beta = float("inf"), depth = 0):

        #picks a move for given state,
        #that will maximize utility for given role,
        #assuming that all other players cooperate
        #to minimize given role's utility
        #and know which move the given role will take

        #returns the move, and the goal value of that move for given role

        sm = self.getStateMachine()

        if sm.isTerminal(state):
            return (sm.getGoal(state, role), None)

        my_moves = sm.getLegalMoves(state, role)

        #instead of using min/max functions,
        #we'll loop through,
        #so we have the option of breaking early,
        #a la alpha/beta search

        chosen_move = None
        best_score = alpha

        for my_move in my_moves:

            #get the move my opponents will make,
            #assuming I make my_move,
            #and they collude against me
            total_moves = sm.getLegalJointMoves(state, role, my_move)
            worst_score = beta

            for total_move in total_moves:
                next_state = sm.getNextState(state, total_move)
                value, _ = self.minmax_move(role, next_state,
                        best_score, worst_score, depth+1)
                worst_score = min(worst_score, value)

                #check if we're currently over-optimizing
                #if our maxnode above us has a superior alpha, that's what's
                #going to get taken anyway
                if worst_score <= best_score:
                    worst_score = best_score
                    break

            #given that they found a move with value 'total_score',
            #should I take this move?
            if depth == 0:
                print "if I pick %r, my value is %r" % (my_move, worst_score)

            if best_score >= beta:
                #can't optimize any better
                return (worst_score, my_move)

            if best_score < worst_score:
                #best score < worst-score
                #so we've definitely improved
                #with this move
                if depth == 0:
                    print "so, I'll pick it"
                best_score = worst_score
                chosen_move = my_move

        return (best_score, chosen_move)

    def stateMachineSelectMove(self, timeout):
        print "gettin some move"

        #as simple heuristic, if only one move available, take it
        moves = self.getStateMachine().getLegalMoves(self.getCurrentState(), self.getRole())
        if len(moves) == 1:
            print "only one move, chosing it"
            return moves[0]
        r, selection = self.minmax_move(self.getRole(), self.getCurrentState())
        print "picked %r with value %r" % (selection, r)
        return selection

    def stateMachineStop(self):
        pass

    def stateMachineAbort(self):
        pass

    def getInitialStateMachine(self):
        return ProverStateMachine()
