from search_problem_lap import *
import open3d as o3d
import numpy as np
import time

TESTING_MODE = True


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
    mesh.compute_vertex_normals() # We need shadows, otherwise the mesh will appear all white
    pcd = o3d.geometry.PointCloud(mesh.vertices)

    list_selected = pick_points(pcd)
    # print(np.asarray(pcd.points)[list_selected])
    points_selected = np.asarray(pcd.points)[list_selected]

    # New Open3D visualizer window...
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Let's cast the selected points to int...
    starting_point = (points_selected[0, :]).astype(int)

    # let's give us 100m of altitude on the goal...
    # We don't want to search for a collision to happen!
    goal_point = (points_selected[1, :]).astype(int)
    goal_point[2] = goal_point[2] + 100

    # Let's make the mesh visible... prepare the line widths to be 10 throughout all the execution...
    o3d.visualization.RenderOption.line_width = 10
    vis.add_geometry(mesh)
    # Start: [27045 11115   148], goal: [ 3555 30735   236]

    if TESTING_MODE:

        starting_point[0] = 27045
        starting_point[1] = 11115
        starting_point[2] = 148

        goal_point[0] = 3555
        goal_point[1] = 30735
        goal_point[2] = 236


    print(f'Start: {starting_point}, goal: {goal_point}')
    print('')
    if TESTING_MODE:
        starting_altitude = 200
        starting_pitch = 0
        starting_yaw = 180
        altitude_limits = 500
    else:
        starting_altitude = int(input(f'Please, give me the starting altitude (note that the point you selected is at {starting_point[2]}m): '))
        starting_pitch = int(input('Please, give me the starting pitch (degrees, [-90, 90] ): '))
        starting_yaw = int(input('Please, give me the starting yaw (degrees, [0, 360[ ): '))
        altitude_limits = int(input('Please, give me the altitude limits: '))
    print('')
    problem = LAPProblem(mesh,
                         LAPState(starting_point[0], starting_point[1], starting_altitude, math.radians(starting_pitch), math.radians(starting_yaw)),
                         LAPState(goal_point[0], goal_point[1], goal_point[2]),
                         altitude_limits, pitch_limits=False)

    src = LAPtreeSearch(enqueueStrategyGreedy(), problem, vis)

    start_time = time.time()
    sol = src.tree_search()
    finish_time = time.time()

    print(f'Solution found in {finish_time - start_time} seconds')
    print(f'Maximum number of nodes generated: {src.statistics_maxNumberOfNodesSeen}')

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
