import open3d as o3d
pcd = o3d.io.read_point_cloud("outputs/pointCloud.ply")
print("hi")
print(pcd)

o3d.visualization.draw_geometries([pcd])