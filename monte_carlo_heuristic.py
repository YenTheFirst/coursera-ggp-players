#from org.ggp.base.util.statemachine.implementation.prover import ProverStateMachine
from org.ggp.base.util.statemachine.implementation.propnet import PropNetStateMachine
from org.ggp.base.player.gamer.statemachine import StateMachineGamer

import time

class PythonHeuristicGamer(StateMachineGamer):

    def getName(self):
        pass

    def stateMachineMetaGame(self, timeout):
        print "running pre-game metagame. must finish within %d" % timeout
        #get an early idea of how long depth charges will take
        self.timeout = timeout/1000

        sm = self.getStateMachine()
        self.depth_time = []
        self.depth_count = []
        init = sm.getInitialState()
        role = self.getRole()

        i = 0
        start_time = time.time()
        while time.time() < self.timeout and i < 100:
            i = i + 1
            self.depth_charge(init, role)
        self.depth_time.append(time.time() - start_time)
        self.depth_count.append(i)
        print self.depth_time, self.depth_count

    def depth_charge(self, state, role):
        sm = self.getStateMachine()
        while not sm.isTerminal(state):
            if time.time() > self.timeout:
                return None
            move = sm.getRandomJointMove(state)
            state = sm.getNextState(state, move)
        #print "   completed a charge..."
        return sm.getGoal(state, role)


    def minmax_move(self, state,
            alpha = float("-inf"), beta = float("inf"),
            depth = 0, expected_moves = 1):

        #picks a move for given state,
        #that will maximize utility for given role,
        #assuming that all other players cooperate
        #to minimize given role's utility
        #and know which move the given role will take

        #returns the move, and the goal value of that move for given role

        sm = self.getStateMachine()
        role = self.getRole()

        #first, if we're terminal, just say so
        if sm.isTerminal(state):
            return (sm.getGoal(state, role), None)

        #we're out of time. just guess at the utility.
        if time.time() > self.timeout:
            print "tight for time! just assuming %r is worth %r-ish" % (my_moves[0], alpha)
            return (alpha, my_moves[0])

        #if we're in deep enough that we should guess instead of exploring
        #further, do so.
        #instead of trying to guess based on depth,
        #we'll use how many moves we expect to examine,
        #and the depth charge calculation time.
        move_count = len(sm.getLegalJointMoves(state))
        time_remaining = self.timeout - time.time()
        charge_count = time_remaining / self.expected_depth_charge_time
        #the charges each move will get if we explore this level
        #and let the next level guess
        charges_per_move = int(charge_count / (move_count * expected_moves))
        #print "%sI expect I have %f s. remaining, and that I have to calculate %d moves. if I expand this level, I'll be able to calculate %d charges per move" % (" "*depth, time_remaining, move_count*expected_moves, charges_per_move)
        #print "%s%f s. %d moves. %d charge per move" % (" "*depth, time_remaining, move_count*expected_moves, charges_per_move)
        if depth > 0 and charges_per_move < 50:
            to_calc = int(charge_count / expected_moves)
            print "%s%d. calculating approx with %d charges" % (" "*depth,depth, to_calc)
            start_time = time.time()
            charges = [self.depth_charge(state, role) for x in xrange(to_calc)]
            charges = [c for c in charges if c is not None]
            if len(charges) == 0:
                return (alpha, None)
            #print " "*depth + "after charge, %r" % charges
            self.depth_time.append(time.time() - start_time)
            self.depth_count.append(to_calc)
            return (sum(charges) / len(charges), None)

        #instead of using min/max functions,
        #we'll loop through,
        #so we have the option of breaking early,
        #a la alpha/beta search

        chosen_move = None
        best_score = alpha

        my_moves = sm.getLegalMoves(state, role)
        moves_examined = 0
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
                        best_score, worst_score,
                        depth+1, (expected_moves * move_count) - moves_examined)
                moves_examined += 1
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
        self.timeout = (timeout / 1000) - 0.5
        start_time = time.time()
        print "gettin some move. must finish by %d. (time now = %d)" % (self.timeout, start_time)

        #as simple heuristic, if only one move available, take it
        moves = self.getStateMachine().getLegalMoves(self.getCurrentState(), self.getRole())
        if len(moves) == 1:
            print "only one move, chosing it"
            return moves[0]

        #update expected runtime for depth charges
        self.depth_time = self.depth_time[-100:]
        self.depth_count = self.depth_count[-100:]
        self.expected_depth_charge_time = sum(self.depth_time) / sum(self.depth_count)
        print "I expect a depth charge will take %f (%d/s)" % (self.expected_depth_charge_time, int(self.expected_depth_charge_time**-1))

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
        #return ProverStateMachine()
        return PropNetStateMachine()
