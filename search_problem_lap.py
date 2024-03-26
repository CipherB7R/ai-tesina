# This is the low altitude search problem implementation
import numpy as np
import open3d as o3d
from utils import *
from search_problems import *


class LAPAction(Action):
    STRAIGHT = 'STRAIGHT'
    LEFT_TURN = 'LEFT TURN'
    RIGHT_TURN = 'RIGHT TURN'
    PULL_UP = 'PULL 4G'
    PULL_DOWN = 'PULL -2G'


class LAPState(State):
    # Information about present position (X,Y,Z)
    def __init__(self, x=0, y=0, h=0, pitch=0, yaw=0):
        super().__init__()
        self.x = x
        self.y = y
        self.h = h
        self.pitch = pitch
        self.yaw = yaw

    def __str__(self):
        return f'({self.x, self.y, self.h}{self.pitch, self.yaw})'


def change_goal_state(goal_x, goal_y, goal_h):
    goal_state.x = goal_x
    goal_state.y = goal_y
    goal_state.z = goal_h


goal_state: LAPState = LAPState(0, 0, 0, 0, 0)


class LAPNode(Node):

    def __init__(self, state: LAPState, action: LAPAction = None, parent=None, path_cost=0., depth=0):
        super().__init__(state, action, parent, path_cost, depth)

    def __lt__(self, other):
        return point3DDistance(self.state.x, self.state.y, self.state.h, goal_state.x, goal_state.y, goal_state.h) < \
            point3DDistance(other.state.x, other.state.y, other.state.h, goal_state.x, goal_state.y, goal_state.h)

    def isEqual(self, other):
        return (self.state.x == other.state.x and
                self.state.y == other.state.y and
                self.state.h == other.state.h and
                self.state.pitch == other.state.pitch
                and self.state.yaw == other.state.yaw)

    def __str__(self):
        return ('' if self.action is None else f'{self.action} --->') + str(self.state)


class LAPProblem(Problem):
    """Class that models the F-16 low altitude ride problem"""
    _TFR_PITCH_LIMIT_MAX = math.radians(40)  # 40 degrees in radians
    _TFR_PITCH_LIMIT_MIN = math.radians(20)  # MINUS 20 degrees in radians

    def __init__(self, terrain_mesh: o3d.geometry.TriangleMesh,
                 initial_state: LAPState,
                 goal_state: LAPState,
                 altitude_limit, goal_distance=100,
                 consider_corners=True,
                 pitch_limits=False,
                 starting_pitch=0,
                 starting_yaw=0,
                 increment=150):
        super().__init__(initial_state, goal_state)
        self.mesh = terrain_mesh
        self.mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        # self.mesh_id = \
        self.scene.add_triangles(self.mesh_legacy)
        change_goal_state(goal_state.x, goal_state.y, goal_state.h)
        self.altitude_limit = altitude_limit
        self.goal_distance = goal_distance
        self.consider_corners = consider_corners
        self.pitch_limits = pitch_limits
        self.increment = increment

    def goal_test(self, state: LAPState):
        return math.sqrt(
            pow(state.x - self.goal_state.x, 2) +
            pow(state.y - self.goal_state.y, 2)
        ) <= self.goal_distance

    def successor_FN(self, state: LAPState) -> set[tuple[LAPState, LAPAction]]:
        # Starting from "state" (a tuple of coordinates) we can only pull 2G.
        # By assuming a standard velocity of 300 knots (154 m/s) and doing some basic physics calculations by
        # equating the centripetal acceleration to the maximum g amount, we obtain that the radius of the turn must be
        #
        #                           r <= 1208,7m
        #
        # Now, our state space is all the possible 3D points which are lower than
        # a determined altitude (barometric) and that are inside the capability of the f-16 turn.
        # We can consider just a fraction of those points, so we can consider a simplified grid.
        #
        # TODO: we will try with increments of 150m.
        #
        # Let's take into account that we need 3 points to draw a circle. In the 3D world, we need 3 points
        # and a plane (identified by those 3 points). That circle, no matter what the plane ( of turn too ) we choose,
        # will need to always have radius <= 1208,7m. Let's say we start the turn in the parent point, continue it in the
        # present state and finish it on the next successor state: this gives us the three points we need to account for.
        #
        # TODO: to reduce the resolution over the X,Y and Z axis, note pulling 2Gs and displacing of 150m permits us
        #  to gain a maximum of 11m in each direction. So, to further restrict the resolution over the z axis we can consider just
        #  the case of pulling 2Gs or not pulling them. This gives us a Grid of 9 points after the current state, distant 150M
        #  from the current state.
        #
        # Now note that we start with a vector in a given point, representing the yaw and pitch at that given point.
        # At every 150m displace we need to consider a new direction, which can modify or not (go straight)
        # the flight vector (pitch and yaw). Let's go!
        successorsSet = set()

        # --------------------------------------------------------------------------------------------
        # Number 1, calculate a list of 9 (5 if self.consider_corners is false) possible displacements
        # --------------------------------------------------------------------------------------------
        # We gain 11m in each direction. We know that an increment of 11m in the y (or x, or z) axis, with r=150m,
        # corresponds to degrees...
        alpha = math.radians(30)
        alpha_down = math.radians(5)
        # alpha = math.atan(11/150)  # TODO: eliminate magic numbers by adding calculation of
        #  max altitude by self.increment and self.max_g
        # phi is the latitude [-pi/2; pi/2], while theta is the xy angle [0; 2pi[.
        # They will modify, respectively, the pitch and yaw.
        # 5 main directions, 4 corners.
        for phi in [-alpha_down, 0, +alpha]:
            for theta in [-alpha, 0, +alpha]:
                if self.consider_corners is False:
                    if abs(phi) == abs(alpha) and (-phi == theta or phi == theta):
                        continue
                    print(f'theta: {newMod((state.yaw + theta), 2 * math.pi)}')
                    new_pitch = newMod((state.pitch + phi), 2 * math.pi) if (state.pitch + phi) >= 0 else (
                            2 * math.pi + (state.pitch + phi))
                    new_yaw = newMod((state.yaw + theta), 2 * math.pi) if (state.yaw + theta) >= 0 else (
                            2 * math.pi + (state.yaw + theta))
                    successorsSet.add(
                        (
                            LAPState(
                                x=math.floor(state.x + self.increment * math.cos(new_pitch) * math.cos(new_yaw)),
                                y=math.floor(state.y + self.increment * math.cos(new_pitch) * math.sin(new_yaw)),
                                h=math.floor(state.h + self.increment * math.sin(new_pitch)),
                                pitch=new_pitch,
                                yaw=new_yaw
                            ),
                            LAPAction.PULL_DOWN  # todo: write more expressive action
                        )
                    )

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
        def heigh_of_mesh(s2: LAPState):
            """S1 will be treated as the origin, s2-s1 as the direction."""

            origin = np.asarray([s2.x, s2.y, 0])
            # d = s2 - s1
            direction = np.asarray([0, 0, 1])

            # print(f'mesh id {mesh_id}')

            # t_hit is the distance to the intersection. The unit is defined by the length of the ray direction.

            od_vector = np.ravel(
                np.row_stack((origin, direction)))  # we need origins and the normalized direction
            # inside the same array, inside the same row
            # (check open3d documentation)

            rays = o3d.core.Tensor([od_vector],
                                   dtype=o3d.core.Dtype.Float32)
            ans = self.scene.cast_rays(rays)
            # print(ans)

            return ans['t_hit'].numpy()[0]

        successorsSet: set[tuple[LAPState, LAPAction]] = {(s, a) for (s, a) in successorsSet if s.h > heigh_of_mesh(s)}

        # --------------------------------------------------------------------------------------------
        # Number 4.2, delete all those that collide with the terrain (mesh)
        # --------------------------------------------------------------------------------------------
        # Create a ray with a direction
        def collides(s1: LAPState, s2: LAPState):
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
            if ans['t_hit'].numpy()[0] <= self.increment:
                # print(f'Collision!!! Distance {ans["t_hit"].numpy()[0]} is inferior to 150.')
                return True
            # print(f'No collision! Distance {ans["t_hit"].numpy()[0]} is superior to 150.')
            return False

        successorsSet: set[tuple[LAPState, LAPAction]] = {(s, a) for (s, a) in successorsSet if not collides(state, s)}

        return successorsSet

    def step_cost(self, n: LAPNode, successor: LAPState) -> float:
        # TODO: TRY with math.floor
        # return point3DDistance(n.state.x, n.state.y, n.state.h,
        #                       successor.x, successor.y, successor.h)
        return self.increment


class enqueueStrategyAstar(enqueueStrategy):
    # TODO: Fails due to too much time!!!

    def calculatePriority(self, node: LAPNode, problem: LAPProblem):
        # print(f'Calculate Priority: {node.path_cost + self.h(node, problem)}')
        return node.path_cost + self.h(node, problem)

    def h(self, node: LAPNode, problem: LAPProblem):
        # heuristic: air line distance
        return point3DDistance(node.state.x, node.state.y, node.state.h,
                               problem.goal_state.x, problem.goal_state.y,
                               problem.goal_state.h)


class enqueueStrategyGreedy(enqueueStrategy):
    # TODO: Fails due to too much time!!!

    def calculatePriority(self, node: LAPNode, problem: LAPProblem):
        # print(f'Calculate Priority: {node.path_cost + self.h(node, problem)}')
        return self.h(node, problem)

    def h(self, node: LAPNode, problem: LAPProblem):
        return point3DDistance(node.state.x, node.state.y, node.state.h,
                               problem.goal_state.x, problem.goal_state.y, problem.goal_state.h)


class enqueueStrategyAstarDynamicWeighting(enqueueStrategy):
    ANTICIPATED_LENGTH = None

    def calculatePriority(self, node: LAPNode, problem: LAPProblem):
        if enqueueStrategyAstarDynamicWeighting.ANTICIPATED_LENGTH is None:
            enqueueStrategyAstarDynamicWeighting.ANTICIPATED_LENGTH = 2 * \
                                                                      point3DDistance(problem.initial_state.x,
                                                                                      problem.initial_state.y,
                                                                                      problem.initial_state.h,
                                                                                      problem.goal_state.x, problem.goal_state.y,
                                                                                      problem.goal_state.h) / problem.increment

        w = 0 if node.depth > enqueueStrategyAstarDynamicWeighting.ANTICIPATED_LENGTH else \
            (1 - node.depth / enqueueStrategyAstarDynamicWeighting.ANTICIPATED_LENGTH)
        fVal = node.path_cost + (1 + 5 * w) * self.h(node, problem)
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

    def debugExpandedNode(self, node: Node):
        # print(f'Visiting newly generated node... Coordinates {node.state.x, node.state.y, node.state.h} with action
        # {node.action}')
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
