from utils import * 

def get_radar(data_path, sample_data, frame_id):
    pc_filename = data_path / sample_data['filename']
    pc = pypcd.PointCloud.from_path(pc_filename)
    msg = numpy_pc2.array_to_pointcloud2(pc.pc_data)
    msg.header.frame_id = frame_id
    msg.header.stamp = get_time(sample_data)
    return msg

def get_lidar(data_path, sample_data, frame_id):
    pc_filename = data_path / sample_data['filename']
    pc_filesize = os.stat(pc_filename).st_size

    with open(pc_filename, 'rb') as pc_file:
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = get_time(sample_data)

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring', offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.is_dense = True
        msg.point_step = len(msg.fields) * 4 # 4 bytes per field
        msg.row_step = pc_filesize
        msg.width = round(pc_filesize / msg.point_step)
        msg.height = 1 # unordered
        msg.data = pc_file.read()
        return msg

def get_camera(data_path, sample_data, frame_id):
    jpg_filename = data_path / sample_data['filename']
    msg = CompressedImage()
    msg.header.frame_id = frame_id
    msg.header.stamp = get_time(sample_data)
    msg.format = "jpeg"
    with open(jpg_filename, 'rb') as jpg_file:
        msg.data = jpg_file.read()
    return msg

def get_camera_info(nusc, sample_data, frame_id):
    calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

    msg_info = CameraInfo()
    msg_info.header.frame_id = frame_id
    msg_info.header.stamp = get_time(sample_data)
    msg_info.height = sample_data['height']
    msg_info.width = sample_data['width']
    msg_info.k[0] = calib['camera_intrinsic'][0][0]
    msg_info.k[1] = calib['camera_intrinsic'][0][1]
    msg_info.k[2] = calib['camera_intrinsic'][0][2]
    msg_info.k[3] = calib['camera_intrinsic'][1][0]
    msg_info.k[4] = calib['camera_intrinsic'][1][1]
    msg_info.k[5] = calib['camera_intrinsic'][1][2]
    msg_info.k[6] = calib['camera_intrinsic'][2][0]
    msg_info.k[7] = calib['camera_intrinsic'][2][1]
    msg_info.k[8] = calib['camera_intrinsic'][2][2]
    
    msg_info.r[0] = 1
    msg_info.r[3] = 1
    msg_info.r[6] = 1
    
    msg_info.p[0] = msg_info.k[0]
    msg_info.p[1] = msg_info.k[1]
    msg_info.p[2] = msg_info.k[2]
    msg_info.p[3] = 0
    msg_info.p[4] = msg_info.k[3]
    msg_info.p[5] = msg_info.k[4]
    msg_info.p[6] = msg_info.k[5]
    msg_info.p[7] = 0
    msg_info.p[8] = 0
    msg_info.p[9] = 0
    msg_info.p[10] = 1
    msg_info.p[11] = 0
    return msg_info