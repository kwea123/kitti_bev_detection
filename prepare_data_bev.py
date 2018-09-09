# BEV cuboids
obj = objects[obj_idx]
box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
corners_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
xmax = np.max(corners_3d_velo[:, 0])
xmin = np.min(corners_3d_velo[:, 0])
ymax = np.max(corners_3d_velo[:, 1])
ymin = np.min(corners_3d_velo[:, 1])
for _ in range(augmentX):
    box2d = np.array([xmin, ymin, xmax, ymax])
    if perturb_box2d:
        xmin,ymin,ymax,ymax = random_shift_box2d(box2d)
    box_fov_inds = (pc_velo[:, 0]<xmax) & (pc_velo[:, 0]>=xmin) & (pc_velo[:, 1]<ymax) & (pc_velo[:, 1]>=ymin)
    box_fov_inds = box_fov_inds & img_fov_inds
    pc_in_box_fov = pc_rect[box_fov_inds,:]

    xc = (xmin+xmax)/2.0
    yc = (ymin+ymax)/2.0
    frustum_angle = -1 * np.arctan2(xc, -yc)

    _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
    label = np.zeros((pc_in_box_fov.shape[0]))
    label[inds] = 1

    heading_angle = obj.ry
    # Get 3D BOX size
    box3d_size = np.array([obj.l, obj.w, obj.h])
    if np.sum(label)==0:
        continue

    id_list.append(data_idx)
    box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
    box3d_list.append(box3d_pts_3d)
    input_list.append(pc_in_box_fov)
    label_list.append(label)
    type_list.append(objects[obj_idx].type)
    heading_list.append(heading_angle)
    box3d_size_list.append(box3d_size)
    frustum_angle_list.append(frustum_angle)

    # collect statistics
    pos_cnt += np.sum(label)
    all_cnt += pc_in_box_fov.shape[0]