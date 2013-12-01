#from org.ggp.base.util.statemachine.implementation.prover import ProverStateMachine
from org.ggp.base.util.statemachine.implementation.propnet import PropNetStateMachine
from org.ggp.base.player.gamer.statemachine import StateMachineGamer
import itertools
import math

import time
import traceback

#represents a state of the game,
#and it's expected utility / confidence on that utility
#it has children for all child states
#each child state relates to a specific joint move

class StateNode(object):
    #class-level cached references to objects
    #will be updated by outside world as necessary
    state_machine = None
    roles = None
    my_role_index = None

    #each state will be a singleton object.
    #the state_cache will be a dict to get at it.
    state_cache = None

    @classmethod
    def get_node(cls, state, parent):
        #gets a node, and sets parent as a parent for the node
        h = state.hashCode()
        if h in cls.state_cache:
            node = cls.state_cache[h]
        else:
            node = cls(state)
            cls.state_cache[state.hashCode()] = node
        node.parents.add(parent)
        return node


    def __init__(self, state):
        self.state = state
        #parents is a list of all parents who point to us
        self.parents = set()

        self.visits = 0

        #children will be a dict of move => child nodes
        #it's initialized when we're explored.
        self.children = None

        #cached score things. to pick where to go and such
        #these will be cached, and updated on exploration / score propagation

        #holds the sum of utility values for all players
        #each player's expected utility is their sum / visits.
        #all utilities will be normalized to [0,1], for easier use in UCB.
        #normalization depends on the max value being 100
        self.utility_sum = [0] * len(StateNode.roles)
        #keeps the same value, normalized to [0,1] and squared, to estimate variance
        self.utility_sum_squared = [0] * len(StateNode.roles)

        #an array of utility UCBs, for each role, for this node.
        #each opponent role will try maxmin this value
        #to pick which child node te explore
        #we will maximize it, based on opponent's maxmin strategies
        self.selection_value = [float('inf')] * len(StateNode.roles)

        #a subset of self.children,
        #that only contains states reachable by the maxmin of opponents
        self.maxmin_children = None

        self.fully_explored = False

    def delete_except(self, to_keep):
        if self.children is not None:
            for child in self.children.values():
                try:
                    child.parents.remove(self)
                except KeyError:
                    print "I'm not in child? what?"
                    print "I'm %r (%r)" % (self.state, self)
                    print "child is %r" % child.state
                    print "parents are %r " % child.parents
                if len(child.parents) == 0 and child != to_keep:
                    child.delete_except(to_keep)
        del StateNode.state_cache[self.state.hashCode()]

    def select_to_explore(self):
        #looks through this node, and child nodes
        #to find which one we should expand & explore

        #if we're unexplored, explore this one.
        if self.children is None:
            return self

        #otherwise, return the best child to explore
        #the best child to explore is the one that
        #gives us the best score
        #out of child states our opponents are likely to pick
        child_to_explore = max(self.maxmin_children.values(),
                key = lambda c: c.selection_value[StateNode.my_role_index])

        #and get their opinion
        return child_to_explore.select_to_explore()

    def update_selection_value(self):
        def sv(util, util_sq):
            parent_visits = sum(p.visits for p in self.parents)
            if self.visits == 0:
                return float("inf")
            if self.fully_explored:
                confidence_range = 0
            else:
                variance = (util_sq / self.visits) - (util / self.visits) ** 2 + math.sqrt(2*math.log(parent_visits) / self.visits)
                confidence_range = min(variance, 0.25) * (math.log(parent_visits) / self.visits)
            return util / self.visits + confidence_range
        self.selection_value = [sv(util, util_sq)
                for util, util_sq in zip(self.utility_sum, self.utility_sum_squared)]

    def update_maxmin_children(self):
        #for each role, keep track of their value for each of their moves
        move_vals = [{} for x in xrange(len(StateNode.roles))]
        for child_move, child in self.children.iteritems():
            for move, score, val in zip(child_move, child.selection_value, move_vals):
                #we only care about the minimum possible score for each move
                #since each role is presumably playing maxmin
                val[move] = min(score, val.get(move, float('inf')))

        #get the maxmin move(s) for each player,
        #based on what the min moves are
        max_vals = [max(v.values()) for v in move_vals]
        maxmin_moves = [[k for k,v in role_mv.iteritems() if v == role_max]
                for role_max, role_mv in zip(max_vals, move_vals)]
        #except, for our role, just include all possible moves
        maxmin_moves[StateNode.my_role_index] = move_vals[StateNode.my_role_index]

        #joint_moves = [list(jm) for jm in itertools.product(*maxmin_moves)]
        #self.maxmin_children = [c for c in self.children if c.move in joint_moves]

        #jython doesn't seem to have itertools.product, so for now we'll do this manually
        self.maxmin_children = {}
        for child_move, child in self.children.iteritems():
            if all((role_move in role_maxmin) for role_move, role_maxmin in zip(child_move, maxmin_moves)):
                self.maxmin_children[child_move] = child

    def update_fully_explored(self):
        #we're fully explored if:
        #a.) all our children are fully explored, or
        #b.) a fully explored child is the minmax

        explored_count = sum(c.fully_explored for c in self.children.values())
        if explored_count == 0:
            return
        if explored_count == len(self.children):
            #all are explored
            self.fully_explored = True
            return

        #else, only a subset explored
        #ok, calculate maxmin
        my_move_vals = {}
        for child in self.children.values():
            my_move_vals[child] = min(child.selection_value, my_move_vals.get(child, child.selection_value))
        max_val = max(my_move_vals.values())
        my_maxmin = [k for k,v in my_move_vals.iteritems() if v == max_val]
        #is our minmax fully explored?
        if any(c.fully_explored for c in my_maxmin):
                self.fully_explored = True

    def explore(self):
        sm = StateNode.state_machine
        #evaluate the state
        if sm.isTerminal(self.state):
            self.fully_explored = True
            score = sm.getGoals(self.state)
        else:
            #do a depthcharge simulation, to get approximate goals
            temp_state = self.state
            while (not sm.isTerminal(temp_state)):
                temp_move = sm.getRandomJointMove(temp_state)
                temp_state = sm.getNextState(temp_state, temp_move)
            score = sm.getGoals(temp_state)

            #add children to the tree
            self.children = {}
            for move in sm.getLegalJointMoves(self.state):
                next_state = sm.getNextState(self.state, move)
                new_node = StateNode.get_node(next_state, self)
                self.children[tuple(move)] = new_node

        return score

    def propagate_score(self, joint_score):
        self.visits += 1
        self.utility_sum = [u + s/100.0 for u,s in zip(self.utility_sum, joint_score)]
        self.utility_sum_squared = [u + (s/100.0)**2 for u,s in zip(self.utility_sum_squared, joint_score)]
        if self.children is not None:
            [c.update_selection_value() for c in self.children.values()]
        if not self.fully_explored:
            self.update_maxmin_children()
            self.update_fully_explored()
        for p in self.parents:
            p.propagate_score(joint_score)

def iterate(root):
    to_explore = root.select_to_explore()
    score = to_explore.explore()
    to_explore.propagate_score(score)


class PythonHeuristicGamer(StateMachineGamer):

    def getName(self):
        pass

    def stateMachineMetaGame(self, timeout):
        print "running pre-game metagame. must finish within %d" % timeout
        #set up the search tree, and get a head start on exploring it
        sm = self.getStateMachine()
        StateNode.state_machine = self.getStateMachine()
        StateNode.roles = sm.getRoles()
        StateNode.my_role_index = StateNode.roles.indexOf(self.getRole())
        StateNode.state_cache={}

        init = self.getStateMachine().getInitialState()
        self.root = StateNode.get_node(init, None)
        self.root.parents = set()

        self.timeout = timeout/1000 - 0.5
        self.iterate_time = 0
        start_time = time.time()

        try:
            iterate_count = self.iterate_until_timeout()
        except:
            traceback.print_exc()

        time_spent = time.time() - start_time
        time_remaining = timeout / 1000 - time.time()
        print "expanded %r iterations in %0.02f seconds (%0.02f remaining). (%0.02f/s)" % (iterate_count, time_spent, time_remaining, iterate_count / time_spent)



    def iterate_until_timeout(self):
        iterate_count = 0
        while time.time() < (self.timeout-self.iterate_time) and not self.root.fully_explored:
            start_time = time.time()
            iterate_count += 1
            explored = iterate(self.root)
            self.iterate_time = time.time() - start_time
        if self.root.fully_explored:
            print "fully explored to the root! now I can kick back and relax"
        return iterate_count

    def stateMachineSelectMove(self, timeout):
        #setup timer & print
        self.timeout = (timeout / 1000) - 0.5
        start_time = time.time()
        print "-"*20
        print "gettin some move. must finish by %d. (time now = %d)" % (self.timeout, start_time)

        #update the search tree.
        most_recent_move = self.getMatch().getMostRecentMoves()
        if most_recent_move is not None:
            most_recent_move = [self.getStateMachine().getMoveFromTerm(t) for t in most_recent_move]
            new_root = self.root.children[tuple(most_recent_move)]

            print "given joint move %r, new root is chosen, with %d/%d visits (%0.02f%%)" % (most_recent_move, new_root.visits, self.root.visits, new_root.visits * 100.0 / self.root.visits)
            #hopefully, original root is now dereferenced,
            #and can be freed from memory. hopefully, python?

            self.root.delete_except(new_root)
            self.root = new_root
            self.root.parents = []

        #calculate up some new best move.
        try:
            iterate_count = self.iterate_until_timeout()
        except:
            traceback.print_exc()

        best_move, best_child = max(self.root.children.iteritems(),
                key = lambda (m,c): c.utility_sum[StateNode.my_role_index])

        #if any of our children have been calculated out to terminal
        #compare them based on win rate, and then pick the real winner
        terminal_children = [(m,c) for m,c in self.root.children.iteritems() if c.fully_explored]
        if len(terminal_children) > 0:
            to_consider = itertools.chain(terminal_children, [(best_move, best_child)])
            best_move, best_child = max(to_consider, key = lambda (m,c): c.utility_sum[StateNode.my_role_index] / c.visits)

        my_move = best_move[StateNode.my_role_index]

        try:
            #try to print some info
            print "right now, what I think about states: "
            for m, c in sorted(self.root.children.iteritems(), key = lambda (m,c): str(m)):
                if m in self.root.maxmin_children:
                    highlight="*"
                else:
                    highlight = ""
                print "%8d %s %r %r %r (fully explored: %r)" % (
                        c.visits, highlight, m,
                        [round(sv,2) for sv in c.selection_value],
                        [round(u/(c.visits+1),2) for u in c.utility_sum],
                        c.fully_explored)

            #print some info
            time_spent = time.time() - start_time
            time_remaining = timeout / 1000 - time.time()

            print "expanded %r iterations in %0.02f seconds (%0.02f remaining). (%0.02f/s)" % (iterate_count, time_spent, time_remaining, iterate_count / time_spent)
            #print "explored %d new visits. %d were repeats (%0.02f)" % (new_visits, repeat_count, repeat_count * 100.0 / iterate_count+1)
            print "picking move %r, with expected value %0.02f" % (my_move, best_child.utility_sum[StateNode.my_role_index] / best_child.visits)
            print "-"*20
        except:
            print "EXCEPTION WITH THE STUPID DEBUG INFO"
            traceback.print_exc()

        return my_move

    def stateMachineStop(self):
        sm = self.getStateMachine()
        print "game has stopped. final goal values: ", sm.getGoals(self.getCurrentState())
        pass

    def stateMachineAbort(self):
        print "game has aborted."
        pass

    def getInitialStateMachine(self):
        #return ProverStateMachine()
        return PropNetStateMachine()
