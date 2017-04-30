"""
This file contains test cases to verify the correct implementation of the
functions required for this project including minimax, alphabeta, and iterative
deepening.  The heuristic function is tested for conformance to the expected
interface, but cannot be automatically assessed for correctness.
STUDENTS SHOULD NOT NEED TO MODIFY THIS CODE.  IT WOULD BE BEST TO TREAT THIS
FILE AS A BLACK BOX FOR TESTING.
"""
import random
import unittest
import timeit
import sys

import isolation
import game_agent

from collections import Counter
from copy import deepcopy
from copy import copy
from functools import wraps
from queue import Queue
from threading import Thread
from multiprocessing import TimeoutError
from queue import Empty as QueueEmptyError
from importlib import reload

WRONG_MOVE = """
The {} function failed because it returned a non-optimal move at search depth {}.
Valid choices: {}
Your selection: {}
"""

WRONG_NUM_EXPLORED = """
Your {} search visited the wrong nodes at search depth {}.  If the number
of visits is too large, make sure that iterative deepening is only
running when the `iterative` flag is set in the agent constructor.
Max explored size: {}
Number you explored: {}
"""

UNEXPECTED_VISIT = """
Your {} search did not visit the number of expected unique nodes at search
depth {}.
Max explored size: {}
Number you explored: {}
"""

ID_FAIL = """
Your agent explored the wrong number of nodes using Iterative Deepening and
minimax. Remember that ID + MM should check every node in each layer of the
game tree before moving on to the next layer.
"""

INVALID_MOVE = """
Your agent returned an invalid move. Make sure that your function returns
a selection when the search times out during iterative deepening.
Valid choices: {!s}
Your choice: {}
"""

TIMER_MARGIN = 15  # time (in ms) to leave on the timer to avoid timeout


def curr_time_millis():
    """Simple timer to return the current clock time in milliseconds."""
    return 1000 * timeit.default_timer()


def handler(obj, testcase, queue):
    """Handler to pass information between threads; used in the timeout
    function to abort long-running (i.e., probably hung) test cases.
    """
    try:
        queue.put((None, testcase(obj)))
    except:
        queue.put((sys.exc_info(), None))


def timeout(time_limit):
    """Function decorator for unittest test cases to specify test case timeout.
    The timer mechanism works by spawning a new thread for the test to run in
    and using the timeout handler for the thread-safe queue class to abort and
    kill the child thread if it doesn't return within the timeout.
    It is not safe to access system resources (e.g., files) within test cases
    wrapped by this timer.
    """

    def wrapUnitTest(testcase):

        @wraps(testcase)
        def testWrapper(self):

            queue = Queue()

            try:
                p = Thread(target=handler, args=(self, testcase, queue))
                p.daemon = True
                p.start()
                err, res = queue.get(timeout=time_limit)
                p.join()
                if err:
                    raise err[0](err[1]).with_traceback(err[2])
                return res
            except QueueEmptyError:
                raise TimeoutError("Test aborted due to timeout. Test was " +
                                   "expected to finish in less than {} second(s).".format(time_limit))

        return testWrapper

    return wrapUnitTest


def makeEvalTable(table):
    """Use a closure to create a heuristic function that returns values from
    a table that maps board locations to constant values. This supports testing
    the minimax and alphabeta search functions.
    THIS HEURISTIC IS ONLY USEFUL FOR TESTING THE SEARCH FUNCTIONALITY -
    IT IS NOT MEANT AS AN EXAMPLE OF A USEFUL HEURISTIC FOR GAME PLAYING.
    """

    def score(game, player):
        row, col = game.get_player_location(player)
        return table[row][col]

    return score


def makeEvalStop(limit, timer, value=None):
    """Use a closure to create a heuristic function that forces the search
    timer to expire when a fixed number of node expansions have been perfomred
    during the search. This ensures that the search algorithm should always be
    in a predictable state regardless of node expansion order.
    THIS HEURISTIC IS ONLY USEFUL FOR TESTING THE SEARCH FUNCTIONALITY -
    IT IS NOT MEANT AS AN EXAMPLE OF A USEFUL HEURISTIC FOR GAME PLAYING.
    """

    def score(game, player):
        if timer.time_left() < 0:
            raise TimeoutError("Timer expired during search. You must " +
                               "return an answer before the timer reaches 0.")
        if limit == game.counts[0]:
            timer.time_limit = 0
        return 0

    return score


def makeBranchEval(first_branch):
    """Use a closure to create a heuristic function that evaluates to a nonzero
    score when the root of the search is the first branch explored, and
    otherwise returns 0.  This heuristic is used to force alpha-beta to prune
    some parts of a game tree for testing.
    THIS HEURISTIC IS ONLY USEFUL FOR TESTING THE SEARCH FUNCTIONALITY -
    IT IS NOT MEANT AS AN EXAMPLE OF A USEFUL HEURISTIC FOR GAME PLAYING.
    """

    def score(game, player):
        if not first_branch:
            first_branch.append(game.root)
        if game.root in first_branch:
            return 1.
        return 0.

    return score


class CounterBoard(isolation.Board):
    """Subclass of the isolation board that maintains counters for the number
    of unique nodes and total nodes visited during depth first search.
    Some functions from the base class must be overridden to maintain the
    counters during search.
    """

    def __init__(self, *args, **kwargs):
        super(CounterBoard, self).__init__(*args, **kwargs)
        self.counter = Counter()
        self.visited = set()
        self.root = None

    def copy(self):
        new_board = CounterBoard(self.__player_1__, self.__player_2__,
                                 width=self.width, height=self.height)
        new_board.move_count = self.move_count
        new_board.__active_player__ = self.__active_player__
        new_board.__inactive_player__ = self.__inactive_player__
        new_board.__last_player_move__ = copy(self.__last_player_move__)
        new_board.__player_symbols__ = copy(self.__player_symbols__)
        new_board.__board_state__ = deepcopy(self.__board_state__)
        new_board.counter = self.counter
        new_board.visited = self.visited
        new_board.root = self.root
        return new_board

    def forecast_move(self, move):
        self.counter[move] += 1
        self.visited.add(move)
        new_board = self.copy()
        new_board.apply_move(move)
        if new_board.root is None:
            new_board.root = move
        return new_board

    @property
    def counts(self):
        """ Return counts of (total, unique) nodes visited """
        return sum(self.counter.values()), len(self.visited)


class MinimaxTest(unittest.TestCase):

    def initAUT(self, depth, score_fn, time_left):
        """Generate and initialize player and board objects to be used for
        testing.
        """
        self.depth = 1
        loc1=(3, 3)
        loc2=(0, 0)
        w = 7
        h = 7
        reload(game_agent)
        agentUT = game_agent.MinimaxPlayer(self.depth, self.score_fn)
        board = CounterBoard(agentUT, 'null_agent', w, h)
        board.apply_move(loc1)
        board.apply_move(loc2)
        return agentUT, board

    # @unittest.skip("Skip simple minimax test.")  # Uncomment this line to skip test

    @timeout(5)
    # @unittest.skip("Skip minimax test.")  # Uncomment this line to skip test
    def test_minimax(self):
        """ Test CustomPlayer.minimax
        This test uses a scoring function that returns a constant value based
        on the location of the search agent on the board to force minimax to
        choose a branch that visits those cells at a specific fixed-depth.
        If minimax is working properly, it will visit a constant number of
        nodes during the search and return one of the acceptable legal moves.
        """
        h, w = 7, 7  # board size
        starting_location = (2, 3)
        adversary_location = (0, 0)  # top left corner
        iterative_search = False


        # The agent under test starts at position (2, 3) on the board, which
        # gives eight (8) possible legal moves [(0, 2), (0, 4), (1, 1), (1, 5),
        # (3, 1), (3, 5), (4, 2), (4, 4)]. The search function will pick one of
        # those moves based on the estimated score for each branch.  The value
        # only changes on odd depths because even depths end on when the
        # adversary has initiative.
        value_table = [[0] * w for _ in range(h)]
        value_table[1][5] = 1  # depth 1 & 2
        value_table[4][3] = 2  # depth 3 & 4
        value_table[6][6] = 3  # depth 5
        heuristic = makeEvalTable(value_table)

        # These moves are the branches that will lead to the cells in the value
        # table for the search depths.
        expected_moves = [set([(1, 5)]),
                          set([(3, 1), (3, 5)]),
                          set([(3, 5), (4, 2)])]

        # Expected number of node expansions during search
        counts = [(8, 8), (24, 10), (92, 27), (418, 32), (1650, 43)]

        # Test fixed-depth search; note that odd depths mean that the searching
        # player (student agent) has the last move, while even depths mean that
        # the adversary has the last move before calling the heuristic
        # evaluation function.
        for idx in range(5):
            test_depth = idx + 1
            agentUT, board = self.initAUT(self, depth, score_fn, time_left,
                                          loc1=starting_location,
                                          loc2=adversary_location)

            # disable search timeout by returning a constant value
            agentUT.time_left = lambda: 1e3
            _, move = agentUT.minimax(board, test_depth)

            num_explored_valid = board.counts[0] == counts[idx][0]
            num_unique_valid = board.counts[1] == counts[idx][1]

            self.assertTrue(num_explored_valid, WRONG_NUM_EXPLORED.format(
                method, test_depth, counts[idx][0], board.counts[0]))

            self.assertTrue(num_unique_valid, UNEXPECTED_VISIT.format(
                method, test_depth, counts[idx][1], board.counts[1]))

            self.assertIn(move, expected_moves[idx // 2], WRONG_MOVE.format(
                method, test_depth, expected_moves[idx // 2], move))

if __name__ == '__main__':
    unittest.main()