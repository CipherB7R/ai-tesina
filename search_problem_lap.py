# This is the low altitude search problem implementation
import numpy as np
import open3d as o3d
from utils import *
from search_problems import *


class LAPAction(Action):
    FORWARD = 'FORWARD'
    BACKWARD = 'BACKWARD'
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'


class LAPState(State):
    # Information about present position (X,Y,Z)
    def __init__(self, x=0, y=0, h=0):
        super().__init__()
        self.x = x
        self.y = y
        self.h = h

    def __str__(self):
        return f'({self.x, self.y, self.h})'


class LAPNode(Node):

    def __init__(self, state: LAPState, action: LAPAction = None, parent=None, path_cost=0., depth=0):
        self.state : LAPState = None
        super().__init__(state, action, parent, path_cost, depth)

    def __lt__(self, other):
        # Overrides the superclass method... Now the preferred node is the one which appears to be the closes to the goal state
        return id(self) < id(other)

    def isEqual(self, other):
        # Overrides the superclass method...
        # Now a node is equal to another if it contains a state with equal coordinates and flight vector.
        return (point3DDistance(self.state.x, self.state.y, self.state.h, other.state.x, other.state.y,
                                other.state.h) < 50)

    def __str__(self):
        # Returns a string containing the action that lead to the current node,
        # plus the stringed view of the state stored in it
        return ('' if self.action is None else f'{self.action} --->') + str(self.state)

"""
return (self.state.x == other.state.x and
                self.state.y == other.state.y and
                self.state.h == other.state.h)
"""



class LAPProblem(Problem):
    """Class that models the AH-64 low altitude ride problem"""

    def __init__(self, terrain_mesh: o3d.geometry.TriangleMesh,
                 initial_state: LAPState,
                 goal_state: LAPState,
                 altitude_limit):
        super().__init__(initial_state, goal_state)

        self.mesh = terrain_mesh

        self.mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(
            self.mesh)  # We need the legacy version for ray casting
        self.scene = o3d.t.geometry.RaycastingScene()
        # self.mesh_id = \  #Debugging purposes for the ray collision calculation, enable this line to save the mesh_id
        self.scene.add_triangles(self.mesh_legacy)

        self.altitude_limit = altitude_limit
        self.goal_distance = 150
        self.increment = 150

        self.previous_action = LAPAction.FORWARD # We enter the simulation by flying straight,
                                                  # already rolled, prepared for the first turn

    def goal_test(self, state: LAPState):
        # The goal test is defined as to check if the state is contained inside a circle of radius self.goal_distance
        return math.sqrt(
            pow(state.x - self.goal_state.x, 2) +
            pow(state.y - self.goal_state.y, 2)
        ) <= self.goal_distance

    def successor_FN(self, state: LAPState) -> set[tuple[LAPState, LAPAction]]:
        def heigh_of_mesh(s2: LAPState):

            # We just need to check the height of the mesh, relative from the XY plane, at the (s2.x,s2.y) position
            origin = np.asarray([s2.x, s2.y, 0])
            direction = np.asarray(
                [0, 0, 1])  # the ray will point upwards, in the same direction of the H (A.K.A. z) axis.

            # we need origins and the normalized direction
            # inside the same array, inside the same row
            # (check open3d documentation)
            od_vector = np.ravel(
                np.row_stack((origin, direction)))

            rays = o3d.core.Tensor([od_vector],
                                   dtype=o3d.core.Dtype.Float32)
            ans = self.scene.cast_rays(rays)
            # print(ans)

            # t_hit is the distance to the intersection (the height of the terrain).
            # The unit is defined by the length of the ray direction, so 1 meter.
            return ans['t_hit'].numpy()[0]

        def collides(s1: LAPState, s2: LAPState, increment):
            """S1 will be treated as the origin, s2-s1 as the direction."""

            origin = np.asarray([s1.x, s1.y, s1.h])
            second_point = np.asarray([s2.x, s2.y, s2.h])
            # d = s2 - s1
            direction = second_point - origin

            # print(f'mesh id {mesh_id}')

            # t_hit is the distance to the intersection. The unit is defined by the length of the ray direction.
            # If there is no intersection this is inf. SO WE NEED TO NORMALIZE THE DIRECTION VECTOR (to obtain UNIT = 1m)
            direction_normalized = direction / np.linalg.norm(direction)
            # print(f'Just checking... direction normal {np.linalg.norm(direction_normalized), direction_normalized}')
            od_vector = np.ravel(
                np.row_stack((origin, direction_normalized)))  # we need origins and the normalized direction
            # inside the same array, inside the same row
            # (check open3d documentation)

            rays = o3d.core.Tensor([od_vector],
                                   dtype=o3d.core.Dtype.Float32)
            ans = self.scene.cast_rays(rays)
            # print(ans)

            # print(ans['t_hit'].numpy(), ans['geometry_ids'].numpy())
            if ans['t_hit'].numpy()[0] <= increment:
                # print(f'Collision!!! Distance {ans["t_hit"].numpy()[0]} is inferior to {increment}.')
                return True
            # print(f'No collision! Distance {ans["t_hit"].numpy()[0]} is superior to {increment}.')
            return False

        successorsSet = set()

        # --------------------------------------------------------------------------------------------
        # Number 1, calculate a list of 5 possible displacements
        # --------------------------------------------------------------------------------------------
        successorsSet.add(
            (
                LAPState(
                    x=(state.x - self.increment),
                    y=(state.y + 0),
                    h=(state.h + 0)
                ),
                LAPAction.LEFT
            )
        )

        successorsSet.add(
            (
                LAPState(
                    x=(state.x + 0),
                    y=(state.y - self.increment),
                    h=(state.h + 0)
                ),
                LAPAction.BACKWARD
            )
        )

        successorsSet.add(
            (
                LAPState(
                    x=(state.x + self.increment),
                    y=(state.y + 0),
                    h=(state.h + 0)
                ),
                LAPAction.RIGHT
            )
        )

        successorsSet.add(
            (
                LAPState(
                    x=(state.x + 0),
                    y=(state.y + self.increment),
                    h=(state.h + 0)
                ),
                LAPAction.FORWARD
            )
        )

        successorsSet.add(
            (
                LAPState(
                    x=(state.x + 0),
                    y=(state.y + 0),
                    h=(state.h - 20)
                ),
                LAPAction.DOWN
            )
        )

        successorsSet.add(
            (
                LAPState(
                    x=(state.x + 0),
                    y=(state.y + 0),
                    h=(state.h + 20)
                ),
                LAPAction.UP
            )
        )


        # --------------------------------------------------------------------------------------------
        # Number 2, delete all those that don't respect the altitude limits
        # --------------------------------------------------------------------------------------------
        successorsSet: set[tuple[LAPState, LAPAction]] = {(s, a) for (s, a) in successorsSet if
                                                          s.h <= self.altitude_limit}

        # --------------------------------------------------------------------------------------------
        # Number 4.1, delete all those that collide with the terrain (mesh)
        # --------------------------------------------------------------------------------------------
        # Check the altitude of the mesh!
        successorsSet: set[tuple[LAPState, LAPAction]] = {(s, a) for (s, a) in successorsSet if s.h > heigh_of_mesh(s)}

        # --------------------------------------------------------------------------------------------
        # Number 4.2, delete all those that collide with the terrain (mesh)
        # --------------------------------------------------------------------------------------------
        # Create a ray with a direction
        successorsSet: set[tuple[LAPState, LAPAction]] = {(s, a) for (s, a) in successorsSet if not collides(state, s, self.increment)}

        return successorsSet

    def step_cost(self, n: LAPNode, successor: LAPState, action: LAPAction) -> float:
        isGoingUpOrDown = action == LAPAction.UP or action == LAPAction.DOWN
        return 20 if isGoingUpOrDown else 150


class enqueueStrategyAstar(enqueueStrategy):
    # Fails due to too much time!!!

    def calculatePriority(self, node: LAPNode, problem: LAPProblem):
        # print(f'Calculate Priority: {node.path_cost + self.h(node, problem)}')
        return node.path_cost + self.h(node, problem)

    def h(self, node: LAPNode, problem: LAPProblem):
        # heuristic: air line distance
        return point3DDistance(node.state.x, node.state.y, node.state.h,
                               problem.goal_state.x, problem.goal_state.y,
                               problem.goal_state.h)


class enqueueStrategyGreedy(enqueueStrategy):

    def calculatePriority(self, node: LAPNode, problem: LAPProblem):
        # print(f'Calculate Priority: {node.path_cost + self.h(node, problem)}')
        return self.h(node, problem)

    def h(self, node: LAPNode, problem: LAPProblem):
        return point3DDistance(node.state.x, node.state.y, node.state.h,
                               problem.goal_state.x, problem.goal_state.y, problem.goal_state.h)


class enqueueStrategyAstarDynamicWeighting(enqueueStrategy):
    # The dynamic weighting approach gives a light weight to the nodes which depth is closest to the expected
    # depth of the solution (measured as the line of air distance divided by the average step of 250m)
    ANTICIPATED_LENGTH = None

    def calculatePriority(self, node: LAPNode, problem: LAPProblem):
        # Calculate the expected depth of solution only one time...
        if enqueueStrategyAstarDynamicWeighting.ANTICIPATED_LENGTH is None:
            enqueueStrategyAstarDynamicWeighting.ANTICIPATED_LENGTH = 2 * \
                                                                      point3DDistance(problem.initial_state.x,
                                                                                      problem.initial_state.y,
                                                                                      problem.initial_state.h,
                                                                                      problem.goal_state.x,
                                                                                      problem.goal_state.y,
                                                                                      problem.goal_state.h) / 250

        # to not assign negative weights, we need to stop when the node depth has surpassed the anticipated length
        w = 0 if node.depth > enqueueStrategyAstarDynamicWeighting.ANTICIPATED_LENGTH else \
            (1 - node.depth / enqueueStrategyAstarDynamicWeighting.ANTICIPATED_LENGTH)

        fVal = node.path_cost + (1 + 3 * w) * self.h(node, problem)
        # print(f'Calculate Priority: {fVal}')
        return fVal

    def h(self, node: LAPNode, problem: LAPProblem):
        # heuristic: air line distance
        return point3DDistance(node.state.x, node.state.y, node.state.h,
                               problem.goal_state.x, problem.goal_state.y,
                               problem.goal_state.h)


class LAPtreeSearch(treeSearch):

    def __init__(self, strategy: enqueueStrategy, problem: LAPProblem, vis: o3d.visualization.Visualizer = None):
        super().__init__(strategy, problem)
        self.vis = vis
        self.statistics_maxNumberOfNodesSeen = 0

    #Got to override it in order to make it work subclass types...
    def expand(self, n: LAPNode, p: LAPProblem) -> set[LAPNode]:
        successors = set()

        for successor, action in p.successor_FN(n.state):
            # print(f'Cost {n.path_cost + p.step_cost(n, successor)}')
            new_n = LAPNode(
                successor,
                action,
                n,
                n.path_cost + p.step_cost(n, successor, action),
                n.depth + 1
            )
            successors.add(new_n)

        return successors

    def afterExpandedNode(self, node: Node):
        # print(f'Visiting newly generated node... Coordinates {node.state.x, node.state.y, node.state.h} with action
        # {node.action}')
        if node.action is None:
            self.previous_action = LAPAction.FORWARD # First node has no action as
                                                      # by defaults of search_problems.treeSearch
        else:
            self.previous_action = node.action

        if len(self.fringe.alreadySeenSet) > self.statistics_maxNumberOfNodesSeen:
            self.statistics_maxNumberOfNodesSeen = len(self.fringe.alreadySeenSet)

        if node.parent is not None and self.vis is not None:
            # let's visualize the line...

            points = [
                [node.parent.state.x, node.parent.state.y, node.parent.state.h],
                [node.state.x, node.state.y, node.state.h]
            ]
            edges = [
                [0, 1]
            ]
            colors = [[1, 0, 0] for i in range(len(edges))]
            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                            lines=o3d.utility.Vector2iVector(edges))
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(line_set)
            self.vis.poll_events()
            self.vis.update_renderer()
            # input('Continue...')
