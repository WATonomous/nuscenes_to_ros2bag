from utils import *

def write_boxes_imagemarkers(nusc, bag, anns, sample_data, frame_id, topic_ns, stamp):
    # annotation boxes
    collector = Collector()
    _, boxes, camera_intrinsic = nusc.get_sample_data(sample_data['token'])
    for box in boxes:
        c = np.array(nusc.explorer.get_color(box.name)) / 255.0
        box.render(collector, view=camera_intrinsic, normalize=True, colors=(c, c, c))

    marker = ImageMarker()
    marker.header.frame_id = frame_id
    marker.header.stamp = get_time(sample_data)
    marker.ns = 'annotations'
    marker.id = 0
    marker.type = ImageMarker.LINE_LIST
    marker.action = ImageMarker.ADD
    marker.scale = 2.0
    marker.points = [make_point2d(p) for p in collector.points]
    marker.outline_colors = [make_color(c) for c in collector.colors]

    bag.write(topic_ns + '/image_markers_annotations', serialize_message(marker), stamp)

def get_lidar_imagemarkers(nusc, sample_lidar, sample_data, frame_id):
    # lidar image markers in camera frame
    points, coloring, _ = nusc.explorer.map_pointcloud_to_image(
        pointsensor_token=sample_lidar['token'],
        camera_token=sample_data['token'],
        render_intensity=True)
    points = points.transpose()
    coloring = [turbomap(c) for c in coloring]

    marker = ImageMarker()
    marker.header.frame_id = frame_id
    marker.header.stamp = get_time(sample_data)
    marker.ns = 'LIDAR_TOP'
    marker.id = 0
    marker.type = ImageMarker.POINTS
    marker.action = ImageMarker.ADD
    marker.scale = 2.0
    marker.points = [make_point2d(p) for p in points]
    marker.outline_colors = [make_color(c) for c in coloring]
    return marker

def find_closest_lidar(nusc, lidar_start_token, stamp_nsec):
    candidates = []

    next_lidar_token = nusc.get("sample_data", lidar_start_token)["next"]
    while next_lidar_token != "":
        lidar_data = nusc.get("sample_data", next_lidar_token)
        if lidar_data["is_key_frame"]:
            break

        dist_abs = abs(stamp_nsec - to_nano(get_time(lidar_data)))
        candidates.append((dist_abs, lidar_data))
        next_lidar_token = lidar_data["next"]

    if len(candidates) == 0:
        return None

    return min(candidates, key=lambda x: x[0])[1]

def get_marker(nusc, annotation_id, stamp):
    ann = nusc.get('sample_annotation', annotation_id)
    marker_id = int(ann['instance_token'][:4], 16)
    c = np.array(nusc.explorer.get_color(ann['category_name'])) / 255.0

    marker = Marker()
    marker.header.frame_id = 'map'
    marker.header.stamp = stamp
    marker.id = marker_id
    marker.text = ann['instance_token'][:4]
    marker.type = Marker.CUBE
    marker.pose = get_pose(ann)
    marker.frame_locked = True
    marker.scale.x = ann['size'][1]
    marker.scale.y = ann['size'][0]
    marker.scale.z = ann['size'][2]
    marker.color = make_color(c, 0.5)
    return marker