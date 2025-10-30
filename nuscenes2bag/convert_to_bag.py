import numpy as np
try:
    _ = np.float
except AttributeError:
    np.float = float

from utils import * 
from sensor_utils import * 
from annotation_utils import *
from map_utils import * 
from can_utils import *

def write_scene(nusc, nusc_can, scene, output_path):
    scene_name = scene['name']
    data_path = Path(nusc.dataroot)
    log = nusc.get('log', scene['log_token'])
    location = log['location']
    print(f'Loading map "{location}"')
    nusc_map = NuScenesMap(dataroot=data_path, map_name=location)

    print(f'Loading bitmap "{nusc_map.map_name}"')
    image = load_bitmap(nusc_map.dataroot, nusc_map.map_name, "basemap")
    print(f"Loaded {image.shape} bitmap")
    print(f"vehicle is {log['vehicle']}")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_path), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    #define sensor topics  
    create_topics(nusc, scene, writer)

    pbar = tqdm(total=get_num_sample_data(nusc, scene), unit="sample_data", desc=f"{scene['name']} Sample Data", leave=False)

    #loop through smaples
    cur_sample = nusc.get("sample", scene["first_sample_token"])


    can_parsers = [
        [nusc_can.get_messages(scene_name, 'ms_imu'), 0, get_imu_msg],
        [nusc_can.get_messages(scene_name, 'pose'), 0, get_odom_msg],
        [nusc_can.get_messages(scene_name, 'steeranglefeedback'), 0, lambda x: get_basic_can_msg('Steering Angle', x)],
        [nusc_can.get_messages(scene_name, 'vehicle_monitor'), 0, lambda x: get_basic_can_msg('Vehicle Monitor', x)],
        [nusc_can.get_messages(scene_name, 'zoesensors'), 0, lambda x: get_basic_can_msg('Zoe Sensors', x)],
        [nusc_can.get_messages(scene_name, 'zoe_veh_info'), 0, lambda x: get_basic_can_msg('Zoe Vehicle Info', x)],
    ]

    # /map
    stamp = get_time(nusc.get('ego_pose', nusc.get('sample_data', cur_sample['data']['LIDAR_TOP'])['ego_pose_token']))
    map_msg = get_scene_map(nusc, scene, nusc_map, image, stamp)
    writer.write('/map', serialize_message(map_msg), to_nano(stamp))

    # /semantic_map
    centerlines_msg = get_centerline_markers(nusc, scene, nusc_map, stamp)
    writer.write('/semantic_map', serialize_message(centerlines_msg), to_nano(stamp))

    while cur_sample is not None:
        
        sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
        stamp = get_time(ego_pose)


        # write CAN messages to /pose, /odom, and /diagnostics
        can_msg_events = []
        for i in range(len(can_parsers)):
            (can_msgs, index, msg_func) = can_parsers[i]
            while index < len(can_msgs) and to_nano(get_utime(can_msgs[index])) < to_nano(stamp):
                can_msg_events.append(msg_func(can_msgs[index]))
                index += 1
                can_parsers[i][1] = index
        can_msg_events.sort(key = lambda x: to_nano(x[0]))
        for (msg_stamp, topic, msg) in can_msg_events:
            writer.write(topic, serialize_message(msg), to_nano(stamp))

        # publish /tf
        tf_array = get_tfmessage(nusc, cur_sample)
        writer.write('/tf', serialize_message(tf_array), to_nano(stamp))

        # /driveable_area occupancy grid
        write_occupancy_grid(writer, nusc_map, ego_pose, stamp)

        # iterate sensors
        for (sensor_id, sample_token) in cur_sample["data"].items():
            pbar.update(1)
            sample_data = nusc.get("sample_data", sample_token)
            topic = "/" + sensor_id
            if sample_data["sensor_modality"] == "radar":
                    msg = get_radar(data_path, sample_data, sensor_id)
                    writer.write(topic, serialize_message(msg), to_nano(stamp))
            elif sample_data["sensor_modality"] == "lidar":
                    msg = get_lidar(data_path, sample_data, sensor_id)
                    writer.write(topic, serialize_message(msg), to_nano(stamp))
            elif sample_data["sensor_modality"] == "camera":
                    msg = get_camera(data_path, sample_data, sensor_id)
                    writer.write(topic + "/image_rect_compressed", serialize_message(msg), to_nano(stamp))
                    msg = get_camera_info(nusc, sample_data, sensor_id)
                    writer.write(topic + "/camera_info", serialize_message(msg), to_nano(stamp))

            if sample_data['sensor_modality'] == 'camera':
                    msg = get_lidar_imagemarkers(nusc, sample_lidar, sample_data, sensor_id)
                    writer.write(topic + '/image_markers_lidar', serialize_message(msg), to_nano(stamp))
                    write_boxes_imagemarkers(nusc, writer, cur_sample['anns'], sample_data, sensor_id, topic, to_nano(stamp))
        
        # publish /pose
        pose_stamped = get_pose_stamped(stamp)
        writer.write('/pose', serialize_message(pose_stamped), to_nano(stamp))
        
        #publish /gps
        gps = get_gps(location, ego_pose, stamp)
        writer.write('/gps', serialize_message(gps), to_nano(stamp))

        #publish /markers/annotations
        marker_array = MarkerArray()
        for annotation_id in cur_sample['anns']:
            marker = get_marker(nusc, annotation_id, stamp)
            marker_array.markers.append(marker)
        writer.write('/markers/annotations', serialize_message(marker_array), to_nano(stamp))

        # collect all sensor frames after this sample but before the next sample
        non_keyframe_sensor_msgs = []
        for (sensor_id, sample_token) in cur_sample['data'].items():
            topic = '/' + sensor_id

            next_sample_token = nusc.get('sample_data', sample_token)['next']
            while next_sample_token != '':
                next_sample_data = nusc.get('sample_data', next_sample_token)
                # if next_sample_data['is_key_frame'] or get_time(next_sample_data).to_nsec() > next_stamp.to_nsec():
                #     break
                if next_sample_data['is_key_frame']:
                    break

                pbar.update(1)
                if next_sample_data['sensor_modality'] == 'radar':
                    msg = get_radar(data_path, next_sample_data, sensor_id)
                    non_keyframe_sensor_msgs.append((to_nano(msg.header.stamp), topic, msg))
                elif next_sample_data['sensor_modality'] == 'lidar':
                    msg = get_lidar(data_path, next_sample_data, sensor_id)
                    non_keyframe_sensor_msgs.append((to_nano(msg.header.stamp), topic, msg))
                elif next_sample_data['sensor_modality'] == 'camera':
                    msg = get_camera(data_path, next_sample_data, sensor_id)
                    camera_stamp_nsec = to_nano(msg.header.stamp)
                    non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_rect_compressed', msg))

                    msg = get_camera_info(nusc, next_sample_data, sensor_id)
                    non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/camera_info', msg))

                    closest_lidar = find_closest_lidar(nusc, cur_sample["data"]["LIDAR_TOP"], camera_stamp_nsec)
                    if closest_lidar is not None:
                        msg = get_lidar_imagemarkers(nusc, closest_lidar, next_sample_data, sensor_id)
                        non_keyframe_sensor_msgs.append(
                            (
                                to_nano(msg.header.stamp),
                                topic + "/image_markers_lidar",
                                msg,
                            )
                        )

                    # Delete all image markers on non-keyframe camera images
                    # msg = get_remove_imagemarkers(sensor_id, 'LIDAR_TOP', msg.header.stamp)
                    # non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_markers_lidar', msg))
                    # msg = get_remove_imagemarkers(sensor_id, 'annotations', msg.header.stamp)
                    # non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_markers_annotations', msg))

                next_sample_token = next_sample_data['next']

        # sort and publish the non-keyframe sensor msgs
        non_keyframe_sensor_msgs.sort(key=lambda x: x[0])
        for (_, topic, msg) in non_keyframe_sensor_msgs:
            writer.write(topic, serialize_message(msg), to_nano(msg.header.stamp))
        # move to the next sample
        cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None

    
    # start_time = 0
    # for i in range(10):
    #     msg = String()
    #     msg.data = f"Chatter #{i}"
    #     timestamp = start_time + (i * 100)
    #     writer.write(TOPIC_NAME, serialize_message(msg), timestamp)

    del writer

def convert_all(
    output_dir: Path,
    name: str,
    nusc: NuScenes,
    nusc_can: NuScenesCanBus,
    selected_scenes,
):
    nusc.list_scenes()
    for scene in nusc.scene:
        scene_name = scene["name"]
        if selected_scenes is not None and scene_name not in selected_scenes:
            continue
        mcap_name = f"NuScenes-{name}-{scene_name}"
        write_scene(nusc, nusc_can, scene, output_dir / mcap_name)

def main():
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", "-s", nargs="*", help="specific scene(s) to write")
    parser.add_argument("--list-only", action="store_true", help="lists the scenes and exits")
    parser.add_argument(
        "--data-dir",
        "-d",
        default="/work/data",
        help="path to nuscenes data directory",
    )
    parser.add_argument(
        "--dataset-name",
        "-n",
        default=["v1.0-mini"],
        nargs="+",
        help="dataset to convert",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=script_dir / "output",
        help="path to write MCAP files into",
    )
    args = parser.parse_args()
    
    nusc_can = NuScenesCanBus(dataroot=args.data_dir)
    
    for name in args.dataset_name:
        nusc = NuScenes(version=name, dataroot=str(args.data_dir), verbose=True)
        if args.list_only:
            nusc.list_scenes()
            return
        convert_all(args.output_dir, name, nusc, nusc_can, args.scene)
    # write_scene(nusc, nusc_can, nusc.scene[0], str(args.output))


if __name__ == "__main__":
    main()