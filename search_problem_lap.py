# This is the low altitude search problem implementation
import math

import numpy as np
import open3d as o3d
from utils import *
from search_problems import *

_GOAL_DISTANCE = 250
_PDT = 50
_EQUAL_YAW_THRESHOLD = 50 # Degrees


_DISPLACEMENT_TURN = 150
_DISPLACEMENT_STRAIGHT_FOR_TURN = 100

class LAPAction(Action):
    STRAIGHT = 'STRAIGHT'
    LEFT_TURN = 'LEFT TURN'
    RIGHT_TURN = 'RIGHT TURN'
    PULL_UP = 'PULL 4G'
    PULL_DOWN = 'PULL -2G'


class LAPState(State):
    # Information about present position (X,Y,Z), plus flight vector (Pitch and yaw)
    def __init__(self, x=0, y=0, h=0, pitch=0, yaw=0):
        super().__init__()
        self.x = x
        self.y = y
        self.h = h
        self.pitch = pitch
        self.yaw = yaw

    def __str__(self):
        return f'({self.x, self.y, self.h}{self.pitch, self.yaw})'


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
                                other.state.h) < _PDT and
                self.state.pitch == other.state.pitch and
                abs(self.state.yaw - other.state.yaw) <= math.radians(_EQUAL_YAW_THRESHOLD))

    def __str__(self):
        # Returns a string containing the action that lead to the current node,
        # plus the stringed view of the state stored in it
        return ('' if self.action is None else f'{self.action} --->') + str(self.state)

"""
return (self.state.x == other.state.x and
                self.state.y == other.state.y and
                self.state.h == other.state.h and
                self.state.pitch == other.state.pitch
                and self.state.yaw == other.state.yaw)
"""



class LAPProblem(Problem):
    """Class that models the F-16 low altitude ride problem"""
    _TFR_PITCH_LIMIT_MAX = math.radians(40)  # 40 degrees in radians
    _TFR_PITCH_LIMIT_MIN = math.radians(-20)  # MINUS 20 degrees in radians

    def __init__(self, terrain_mesh: o3d.geometry.TriangleMesh,
                 initial_state: LAPState,
                 goal_state: LAPState,
                 altitude_limit,
                 pitch_limits=True):
        super().__init__(initial_state, goal_state)

        self.mesh = terrain_mesh

        self.mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(
            self.mesh)  # We need the legacy version for ray casting
        self.scene = o3d.t.geometry.RaycastingScene()
        # self.mesh_id = \  #Debugging purposes for the ray collision calculation, enable this line to save the mesh_id
        self.scene.add_triangles(self.mesh_legacy)

        self.altitude_limit = altitude_limit
        self.goal_distance = _GOAL_DISTANCE  # meters, radius of the "goal circle"
        self.pitch_limits = pitch_limits

        self.previous_action = LAPAction.STRAIGHT  # We enter the simulation by flying straight,
        # already rolled, prepfared for the first turn

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

        def new_state(starting_state, displacement, deltaPitch, deltaYaw):
            # takes in the starting state, the displacement and the delta angles... Gives back a new state with new position!
            new_pitch = newMod((starting_state.pitch + deltaPitch), 2 * math.pi) if (
                                                                                            starting_state.pitch + deltaPitch) >= 0 else (
                    2 * math.pi + (starting_state.pitch + deltaPitch))
            new_yaw = newMod((starting_state.yaw + deltaYaw), 2 * math.pi) if (
                                                                                      starting_state.yaw + deltaYaw) >= 0 else (
                    2 * math.pi + (starting_state.yaw + deltaYaw))

            return LAPState(
                x=math.floor(starting_state.x + displacement * math.cos(new_pitch) * math.cos(new_yaw)),
                y=math.floor(starting_state.y + displacement * math.cos(new_pitch) * math.sin(new_yaw)),
                h=math.floor(starting_state.h + displacement * math.sin(new_pitch)),
                pitch=new_pitch,
                yaw=new_yaw
            )

        # New empty successor set...
        successorsSet = set()

        # --------------------------------------------------------------------------------------------
        # Number 1, calculate a list of 5 possible displacements
        # --------------------------------------------------------------------------------------------

        # A) 100m displacement
        state_100m_displacement = new_state(state, _DISPLACEMENT_STRAIGHT_FOR_TURN, 0, 0)

        # B) 100m displacement collision test
        can_change_direction = not collides(state, state_100m_displacement,
                                            _DISPLACEMENT_STRAIGHT_FOR_TURN) and state_100m_displacement.h > heigh_of_mesh(state_100m_displacement)

        deltaTheta = math.radians(15)
        deltaThetaDown = math.radians(6.5)

        if can_change_direction:
            # 5 possible directions!

            # 1) we can go straight, whatever the previous action was (WITHOUT THE 100M displacement!).
            temp_state = \
                (
                    new_state(state, _DISPLACEMENT_TURN, 0, 0),
                    LAPAction.STRAIGHT
                )
            successorsSet.add(temp_state) if not collides(state, temp_state[0], _DISPLACEMENT_TURN) else None

            # 2) we can go up, and, in the case we were already going up (or straight) we can continue without THE 100M DISPLACEMENT
            if self.previous_action == LAPAction.STRAIGHT or self.previous_action == LAPAction.PULL_UP:
                temp_state = \
                    (
                        new_state(state, _DISPLACEMENT_TURN, deltaTheta, 0),
                        LAPAction.PULL_UP
                    )
                successorsSet.add(temp_state) if not collides(state, temp_state[0], _DISPLACEMENT_TURN) else None
            else:
                temp_state = \
                    (
                        new_state(state_100m_displacement, _DISPLACEMENT_TURN, deltaTheta, 0),
                        LAPAction.PULL_UP
                    )
                successorsSet.add(temp_state) if not collides(state_100m_displacement, temp_state[0], _DISPLACEMENT_TURN) else None

            # 3) we can go down, and, in the case we were already going down (or straight) we can continue without THE 100M DISPLACEMENT
            if self.previous_action == LAPAction.STRAIGHT or self.previous_action == LAPAction.PULL_DOWN:
                temp_state = \
                    (
                        new_state(state, _DISPLACEMENT_TURN, -deltaThetaDown, 0),
                        LAPAction.PULL_DOWN
                    )
                successorsSet.add(temp_state) if not collides(state, temp_state[0], _DISPLACEMENT_TURN) else None
            else:
                temp_state = \
                    (
                        new_state(state_100m_displacement, _DISPLACEMENT_TURN, -deltaThetaDown, 0),
                        LAPAction.PULL_DOWN
                    )
                successorsSet.add(temp_state) if not collides(state_100m_displacement, temp_state[0], _DISPLACEMENT_TURN) else None
            # 4) we can go left, and, in the case we were already going left (or straight) we can continue without THE 100M DISPLACEMENT
            if self.previous_action == LAPAction.STRAIGHT or self.previous_action == LAPAction.LEFT_TURN:
                temp_state = \
                    (
                        new_state(state, _DISPLACEMENT_TURN, 0, deltaTheta),
                        LAPAction.LEFT_TURN
                    )
                successorsSet.add(temp_state) if not collides(state, temp_state[0], _DISPLACEMENT_TURN) else None
            else:
                temp_state = \
                    (
                        new_state(state_100m_displacement, _DISPLACEMENT_TURN, 0, deltaTheta),
                        LAPAction.LEFT_TURN
                    )
                successorsSet.add(temp_state) if not collides(state_100m_displacement, temp_state[0], _DISPLACEMENT_TURN) else None

            # 5) we can go right, and, in the case we were already going right (or straight) we can continue without THE 100M DISPLACEMENT
            if self.previous_action == LAPAction.STRAIGHT or self.previous_action == LAPAction.RIGHT_TURN:
                temp_state = \
                    (
                        new_state(state, _DISPLACEMENT_TURN, 0, -deltaTheta),
                        LAPAction.RIGHT_TURN
                    )
                successorsSet.add(temp_state) if not collides(state, temp_state[0], _DISPLACEMENT_TURN) else None
            else:
                temp_state = \
                    (
                        new_state(state_100m_displacement, _DISPLACEMENT_TURN, 0, -deltaTheta),
                        LAPAction.RIGHT_TURN
                    )
                successorsSet.add(temp_state) if not collides(state_100m_displacement, temp_state[0], _DISPLACEMENT_TURN) else None
        else:
            # 4 possible directions! (This time without the option to abruptly change course!)

            # 1) we can't go straight, THE COLLISION CHECK FAILED FOR 100m, OF COURSE IT WILL FAIL FOR 150m TOO!

            # 2) we can go up in the case we were already going up (or straight), hoping that we can advert the obstacle!
            if self.previous_action == LAPAction.STRAIGHT or self.previous_action == LAPAction.PULL_UP:
                temp_state = \
                    (
                        new_state(state, _DISPLACEMENT_TURN, deltaTheta, 0),
                        LAPAction.PULL_UP
                    )
                successorsSet.add(temp_state) if not collides(state, temp_state[0], _DISPLACEMENT_TURN) else None

            # 3) we can go down in the case we were already going down (or straight), hoping that we can advert the obstacle!
            if self.previous_action == LAPAction.STRAIGHT or self.previous_action == LAPAction.PULL_DOWN:
                temp_state = \
                    (
                        new_state(state, _DISPLACEMENT_TURN, -deltaThetaDown, 0),
                        LAPAction.PULL_DOWN
                    )
                successorsSet.add(temp_state) if not collides(state, temp_state[0], _DISPLACEMENT_TURN) else None

            # 4) we can go left in the case we were already going left (or straight), hoping that we can advert the obstacle!
            if self.previous_action == LAPAction.STRAIGHT or self.previous_action == LAPAction.LEFT_TURN:
                temp_state = \
                    (
                        new_state(state, _DISPLACEMENT_TURN, 0, deltaTheta),
                        LAPAction.LEFT_TURN
                    )
                successorsSet.add(temp_state) if not collides(state, temp_state[0], _DISPLACEMENT_TURN) else None

            # 5) we can go right in the case we were already going right (or straight) hoping that we can advert the obstacle!
            if self.previous_action == LAPAction.STRAIGHT or self.previous_action == LAPAction.RIGHT_TURN:
                temp_state = \
                    (
                        new_state(state, _DISPLACEMENT_TURN, 0, -deltaTheta),
                        LAPAction.RIGHT_TURN
                    )
                successorsSet.add(temp_state) if not collides(state, temp_state[0], _DISPLACEMENT_TURN) else None

        # --------------------------------------------------------------------------------------------
        # Number 2, delete all those that don't respect the altitude limits
        # --------------------------------------------------------------------------------------------
        successorsSet: set[tuple[LAPState, LAPAction]] = {(s, a) for (s, a) in successorsSet if
                                                          s.h <= self.altitude_limit}

        # --------------------------------------------------------------------------------------------
        # Number 3, delete all those that don't respect the pitch limits (Optional)
        # --------------------------------------------------------------------------------------------
        if self.pitch_limits:
            successorsSet = {(s, a) for (s, a) in successorsSet if
                             self._TFR_PITCH_LIMIT_MIN <= s.pitch <= self._TFR_PITCH_LIMIT_MAX}

        # --------------------------------------------------------------------------------------------
        # Number 4.1, delete all those that collide with the terrain (mesh)
        # --------------------------------------------------------------------------------------------
        # Check the altitude of the mesh!
        successorsSet: set[tuple[LAPState, LAPAction]] = {(s, a) for (s, a) in successorsSet if s.h > heigh_of_mesh(s)}

        return successorsSet

    def step_cost(self, n: LAPNode, successor: LAPState, action: LAPAction) -> float:
        isStraight = action == LAPAction.STRAIGHT
        isSameAction = n.action == action
        turnOrPullAfterStraight = n.action == LAPAction.STRAIGHT and (
                action is LAPAction.PULL_UP or
                action is LAPAction.PULL_DOWN or
                action is LAPAction.LEFT_TURN or
                action is LAPAction.RIGHT_TURN
        )
        return _DISPLACEMENT_TURN if isStraight or isSameAction or turnOrPullAfterStraight else _DISPLACEMENT_TURN + _DISPLACEMENT_STRAIGHT_FOR_TURN


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
                                                                                      problem.goal_state.h) / (_DISPLACEMENT_TURN + _DISPLACEMENT_STRAIGHT_FOR_TURN)

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
            # First node has no action as by defaults of search_problems.treeSearch
            # We need to assign it ourselves...
            self.previous_action = LAPAction.STRAIGHT
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
