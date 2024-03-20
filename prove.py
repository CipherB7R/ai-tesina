import open3d as o3d
import copy
import numpy as np

mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh('SK_pezzettoHD.stl')
vis = o3d.visualization.Visualizer()
vis.create_window()
mesh.compute_vertex_normals()

# I = np.eye(4)
# I[2,2] = -1
# mesh_mirror = copy.deepcopy(mesh).transform(I)
# mesh_mirror.compute_vertex_normals()

# print(np.flip(np.asarray(mesh.triangles)))
#new_mesh = o3d.geometry.TriangleMesh()

origin = np.asarray([150, 150, 120])
second_point = np.asarray([150, 150, 270])
direction = second_point - origin
points = [
    origin,
    origin + direction
]
edges = [
    [0, 1]
]
colors = [[1, 0, 0] for i in range(len(edges))]

line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(edges))

line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.RenderOption.line_width = 10
vis.add_geometry(mesh)
vis.add_geometry(line_set)

scene = o3d.t.geometry.RaycastingScene()
mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
mesh_id = scene.add_triangles(mesh_legacy)

print(f'mesh id {mesh_id}')

# t_hit is the distance to the intersection. The unit is defined by the length of the ray direction.
# If there is no intersection this is inf. SO WE NEED TO NORMALIZE THE DIRECTION VECTOR (to obtain UNIT = 1m)
direction_normalized = direction / np.linalg.norm(direction)
print(f'Just checking... direction normal {np.linalg.norm(direction_normalized), direction_normalized}')
od_vector = np.ravel(np.row_stack((origin, direction_normalized))) # we need origins and the normalized direction
                                                       # inside the same array, inside the same row
                                                       # (check open3d documentation)

rays = o3d.core.Tensor([od_vector],
                       dtype=o3d.core.Dtype.Float32)

ans = scene.cast_rays(rays)
#print(ans)
if ans['t_hit'].numpy()[0] <= 150:
    print(f'Collision!!! Distance {ans["t_hit"].numpy()[0]} is inferior to 150.')
else:
    print(f'No collision! Distance {ans["t_hit"].numpy()[0]} is superior to 150.')

print(ans['t_hit'].numpy(), ans['geometry_ids'].numpy())



vis.run()
