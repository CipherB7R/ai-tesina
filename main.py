from search_problem_lap import *
import open3d as o3d
import numpy as np


def pick_points(pcd):
    print("")
    print(
        "1) Please pick two points, the starting and goal ones, using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


if __name__ == '__main__':
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh('SK_pezzettoHD.stl')
    mesh.compute_vertex_normals()
    # print(mesh.get_center())
    pcd = o3d.geometry.PointCloud(mesh.vertices)

    list_selected = pick_points(pcd)
    # print(np.asarray(pcd.points)[list_selected])

    points_selected = np.asarray(pcd.points)[list_selected]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    starting_point = (points_selected[0, :]).astype(int)
    starting_point[2] = starting_point[2] + 100  # let's give us 100m of altitude

    goal_point = (points_selected[1, :]).astype(int)
    goal_point[2] = goal_point[2] + 100  # let's give us 100m of altitude

    points = [
        starting_point,
        goal_point
    ]

    # Visualize the path (air line, may collide...)
    edges = [
        [0, 1]
    ]
    colors = [[1, 0, 0] for i in range(len(edges))]

    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(edges))

    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.RenderOption.line_width = 10
    vis.add_geometry(mesh)

    print(f'Start: {starting_point}, goal: {goal_point}')
    print('')
    starting_altitude = int(input(f'Please, give me the starting altitude (note that the point you selected is at {starting_point[2]}m): '))
    starting_pitch = int(input('Please, give me the starting pitch (degrees, [-90, 90] ): '))
    starting_yaw = int(input('Please, give me the starting roll (degrees, [0, 360[ ): '))
    altitude_limits = int(input('Please, give me the altitude limits: '))
    print('')
    problem = LAPProblem(mesh,
                         LAPState(starting_point[0], starting_point[1], starting_altitude, math.radians(starting_pitch), math.radians(starting_yaw)),
                         LAPState(goal_point[0], goal_point[1], goal_point[2]),
                         altitude_limits, pitch_limits=False)

    src = LAPtreeSearch(enqueueStrategyAstarDynamicWeighting(), problem, vis)

    sol = src.tree_search()


    input('Waiting to continue... Will clear the Open3D visualizer and show the solution!')

    vis.clear_geometries()
    vis.add_geometry(mesh)
    if sol is not None:
        for n in sol:
            print(n.state.__str__())
            if n.parent is not None:
                points = [
                    [n.parent.state.x, n.parent.state.y, n.parent.state.h],
                    [n.state.x, n.state.y, n.state.h]
                ]
                edges = [
                    [0, 1]
                ]
                colors = [[0, 1, 0] for i in range(len(edges))]
                line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                                lines=o3d.utility.Vector2iVector(edges))
                line_set.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_set)
                vis.poll_events()
                vis.update_renderer()





    vis.run()
    # vis.update_renderer()
