#from org.ggp.base.util.statemachine.implementation.prover import ProverStateMachine
from org.ggp.base.util.statemachine.implementation.prover import ProverStateMachine
from org.ggp.base.player.gamer.statemachine import StateMachineGamer

import time

class PythonHeuristicGamer(StateMachineGamer):

    def getName(self):
        pass

    def stateMachineMetaGame(self, timeout):
        print "running pre-game metagame. must finish within %d" % timeout

    def depth_charge(self, state, role):
        sm = self.getStateMachine()
        while not sm.isTerminal(state):
            if time.time() > self.timeout:
                return None
            move = sm.getRandomJointMove(state)
            state = sm.getNextState(state, move)
        print "   completed a charge..."
        return sm.getGoal(state, role)


    def minmax_move(self, state,
            alpha = float("-inf"), beta = float("inf"), depth = 0):

        #picks a move for given state,
        #that will maximize utility for given role,
        #assuming that all other players cooperate
        #to minimize given role's utility
        #and know which move the given role will take

        #returns the move, and the goal value of that move for given role

        sm = self.getStateMachine()
        role = self.getRole()

        if sm.isTerminal(state):
            return (sm.getGoal(state, role), None)
        if depth > 1:
            charges = [self.depth_charge(state, role) for x in xrange(3)]
            charges = [c for c in charges if c is not None]
            if len(charges) == 0:
                return (alpha, None)
            print " "*depth + "after charge, %r" % charges
            return (sum(charges) / len(charges), None)


        my_moves = sm.getLegalMoves(state, role)

        if time.time() > self.timeout:
            #we're out of time. just guess at the utility.
            print "tight for time! just assuming %r is worth %r-ish" % (my_moves[0], alpha)
            return (alpha, my_moves[0])



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
                print " "*depth + "%d. considering %r" % (depth, total_move)
                next_state = sm.getNextState(state, total_move)
                value, _ = self.minmax_move(next_state,
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

        return (best_score, chosen_move or my_moves[0])


    def stateMachineSelectMove(self, timeout):
        self.timeout = timeout / 1000
        start_time = time.time()
        print "gettin some move. must finish by %d. (time now = %d)" % (self.timeout, start_time)

        #as simple heuristic, if only one move available, take it
        moves = self.getStateMachine().getLegalMoves(self.getCurrentState(), self.getRole())
        if len(moves) == 1:
            print "only one move, chosing it"
            return moves[0]
        r, selection = self.minmax_move(self.getCurrentState())
        time_remaining = int(self.timeout - time.time())
        time_spent = time.time() - start_time
        print "picked %r with value %r,took %d seconds, with %d seconds to spare" %\
                (selection, r, time_spent, time_remaining)
        return selection

    def stateMachineStop(self):
        print "game has stopped. final goal values: "
        pass

    def stateMachineAbort(self):
        print "game has aborted."
        pass

    def getInitialStateMachine(self):
        return ProverStateMachine()
