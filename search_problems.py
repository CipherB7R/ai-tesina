# This is the search problems abstract structure
import abc
from enum import Enum
from queue import PriorityQueue
from typing import Union


class Action(Enum):
    pass

class State(abc.ABC):
    pass


class Node:

    def __init__(self, state: State, action: Action = None, parent=None, path_cost=0., depth=0):
        self.state = state
        self.action = action
        self.parent = parent
        self.path_cost = path_cost
        self.depth = depth

    def __lt__(self, other):
        # This is just for compatibility purposes with the PriorityQueue class.
        # In fact, PriorityQueue accepts elements which are tuples (priority, node) and we
        # need to provide a method to discern the location even when priority are equal between nodes!
        # Feel free to override this method!
        return id(self) < id(other)

    def isEqual(self, other):
        return self == other

    def solution(self):
        # Return the ordered list of all the parents of this node.
        # The list should be reversed in order to obtain the path from
        # the starting state to the final state (solution)
        # This function does not do this.
        if self.parent is None:
            return [self]
        else:
            return [self] + self.parent.solution()

class Problem(abc.ABC):

    def __init__(self, initial_state: State, goal_state: State):
        self.initial_state = initial_state
        self.goal_state = goal_state

    @abc.abstractmethod
    def goal_test(self, state: State):
        pass

    @abc.abstractmethod
    def successor_FN(self, state: State) -> set[tuple[State, Action]]:
       pass

    @abc.abstractmethod
    def step_cost(self, n: Node, successor: State, action: Action) -> float:
        pass



class enqueueStrategy(abc.ABC):
    """
    All subclasses will implement a different enqueue strategy, namely a different
    search algorythm.
    """

    @abc.abstractmethod
    def calculatePriority(self, node: Node, problem: Problem):
        return 1  # Base strategy implements a simple FIFO queue


class NodePriorityQueue:

    def __init__(self, problem: Problem, strategy: enqueueStrategy):
        self._queue = PriorityQueue()
        self.alreadySeenSet = set()
        self._problem = problem
        self.strategy = strategy

    def removeFirst(self):
        temp = self._queue.get()
        # print(f'Remove First: {temp[0]}')
        return temp[1]

    def enqueue(self, n: Union[set[Node], Node]):
        if isinstance(n, Node):
            already_saved = False

            priority_of_node = self.strategy.calculatePriority(n, self._problem)
            n.priority = priority_of_node

            for p, savedNode in self.alreadySeenSet:
                if n.isEqual(savedNode):
                    #print('already seen node')
                    if priority_of_node < p:
                        #print('Replaced a node with a better one')
                        # Replace those that have higher priority with those with lower one...
                        self.alreadySeenSet.remove((p, savedNode))
                        self.alreadySeenSet.add((n.priority, n))
                        if (p,savedNode) in self._queue.queue:
                            self._queue.queue.remove((p, savedNode))
                            self._queue.put((n.priority, n))
                    # Ignore already seen nodes
                    already_saved = True
                    break

            if already_saved is False:
                self._queue.put((n.priority, n))
                self.alreadySeenSet.add((n.priority, n))

        else:
            for e in n:
                self.enqueue(e)

    def empty(self):
        return self._queue.empty()




class treeSearch:

    def __init__(self, strategy: enqueueStrategy, problem: Problem):
        self.fringe = None
        self.problem = problem
        self.strategy = strategy

    def tree_search(self):
        self.fringe = NodePriorityQueue(self.problem, self.strategy)
        self.fringe.enqueue(
            Node(self.problem.initial_state)
        )

        while True:
            if self.fringe.empty():
                print("Failure")
                return None

            node: Node = self.fringe.removeFirst()  # Same as REMOVE-FIRST(q)

            self.afterExpandedNode(node)

            if self.problem.goal_test(node.state):
                print('Solution found!')
                return node.solution()

            self.fringe.enqueue(self.expand(node, self.problem))

    def expand(self, n: Node, p: Problem) -> set[Node]:
        successors = set()

        for successor, action in p.successor_FN(n.state):
            # print(f'Cost {n.path_cost + p.step_cost(n, successor)}')
            new_n = Node(
                successor,
                action,
                n,
                n.path_cost + p.step_cost(n, successor, action),
                n.depth + 1
            )
            successors.add(new_n)

        return successors

    def afterExpandedNode(self, node: Node):
        pass


