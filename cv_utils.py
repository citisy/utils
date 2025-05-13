"""utils for computer vision task"""
import cv2
import numpy as np


class CoordinateConvert:
    @staticmethod
    def _call(bbox, wh, blow_up, convert_func):
        tmp_bbox = np.array(bbox)
        flag = len(tmp_bbox.shape) == 1

        bbox = np.array(bbox).reshape((-1, 4))
        convert_bbox = np.zeros_like(bbox)

        if wh is None:
            wh = (1, 1)

        wh = np.array(wh)

        if not blow_up:
            wh = 1 / wh

        wh = np.r_[wh, wh]

        convert_bbox = convert_func(bbox, convert_bbox) * wh

        if flag:
            convert_bbox = convert_bbox[0]

        return convert_bbox

    @classmethod
    def mid_xywh2top_xyxy(cls, bbox, wh=None, blow_up=True):
        """中心点xywh转换成顶点xyxy

        Args:
            bbox: xywh, xy, middle coordinate, wh, width and height of object
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xyxy(tuple): 左上右下顶点xy坐标
        """

        def convert_func(bbox, convert_bbox):
            convert_bbox[:, 0:2] = bbox[:, 0:2] - bbox[:, 2:4] / 2
            convert_bbox[:, 2:4] = bbox[:, 0:2] + bbox[:, 2:4] / 2
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xywh2top_xyxy(cls, bbox, wh=None, blow_up=True):
        """顶点xywh转换成顶点xyxy

        Args:
            bbox: xywh, xy, left top coordinate, wh, width and height of object
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xyxy(tuple): 左上右下顶点xy坐标
        """

        def convert_func(bbox, convert_bbox):
            convert_bbox[:, 0:2] = bbox[:, 0:2]
            convert_bbox[:, 2:4] = bbox[:, 0:2] + bbox[:, 2:4]
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xywh2mid_xywh(cls, bbox, wh=None, blow_up=True):
        """顶点xywh转中心点xywh

        Args:
            bbox: xywh, xy, left top coordinate, wh, width and height of object
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xywh(tuple): xy -> 中心点坐标, wh -> 目标宽高
        """

        def convert_func(bbox, convert_bbox):
            convert_bbox[:, 0:2] = bbox[:, 0:2] + bbox[:, 2:4] / 2
            convert_bbox[:, 2:4] = bbox[:, 2:4]
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xyxy2top_xywh(cls, bbox, wh=None, blow_up=True):
        """顶点xyxy转顶点xywh

        Args:
            bbox: xyxy, left top and right down
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xywh(tuple): xy -> 左上顶点坐标, wh -> 目标宽高
        """

        def convert_func(bbox, convert_bbox):
            convert_bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, 0:2]
            convert_bbox[:, 0:2] = bbox[:, 0:2]
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xyxy2mid_xywh(cls, bbox, wh=None, blow_up=True):
        """顶点xyxy转换成中心点xywh

        Args:
            bbox: xyxy, left top and right down
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xywh(tuple): xy -> 中心点点坐标, wh -> 目标宽高
        """

        def convert_func(bbox, convert_bbox):
            convert_bbox[:, 0:2] = (bbox[:, 0:2] + bbox[:, 2:4]) / 2
            convert_bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, 0:2]
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @staticmethod
    def rect2box(rects) -> np.ndarray:
        """rects(-1, 4, 2) convert to boxes(-1, 4)"""
        rects = np.array(rects)

        if rects.size == 0:
            return np.zeros((0, 4))

        x1 = np.min(rects[:, :, 0], axis=1)
        y1 = np.min(rects[:, :, 1], axis=1)
        x2 = np.max(rects[:, :, 0], axis=1)
        y2 = np.max(rects[:, :, 1], axis=1)
        boxes = np.c_[x1, y1, x2, y2]
        return boxes

    @staticmethod
    def box2rect(boxes) -> np.ndarray:
        """boxes(-1, 4) convert to rects(-1, 4, 2)"""
        boxes = np.array(boxes)

        if boxes.size == 0:
            return np.zeros((0, 0, 2))

        rects = np.zeros((len(boxes), 4, 2))
        rects[:, 0] = boxes[:, :2]
        rects[:, 1] = boxes[:, (2, 1)]
        rects[:, 2] = boxes[:, 2:]
        rects[:, 3] = boxes[:, (0, 3)]
        return rects


def detect_continuous_lines(image, tol=0, region_thres=0, binary_thres=200, axis=1):
    """detect vertical or horizontal lines which have continuous pixels

    Args:
        image: 3-D array(h, w, c) or 2-D array(h, w)
        tol(int): num of blank pixels lower than tol will be treated as one line
        region_thres(int): filter lines whose length is lower than region_thres
        binary_thres(int): binary images threshold, fall in [0, 255]
        axis: 0 for y-axis lines, 1 for x-axis lines

    Returns:
        lines: 2-D array, (m, 2)
    """
    if len(image.shape) == 3:
        # binary
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, binary_thres, 1, cv2.THRESH_BINARY_INV)

    projection = np.any(image, axis=axis)

    projection = np.insert(projection, 0, 0)
    projection = np.append(projection, 0)
    diff = np.diff(projection)
    start = np.argwhere(diff == 1).flatten()
    end = np.argwhere(diff == -1).flatten() - 1
    lines = np.stack((start, end), axis=1)

    idx = np.where((np.abs(lines[1:, 0] - lines[:-1, 1])) < tol)[0]

    for i in idx[::-1]:
        lines[i, 1] = lines[i + 1, 1]

    flag = np.ones(len(lines), dtype=bool)
    flag[idx + 1] = False
    lines = lines[flag]

    # length larger than region_thres
    lines = lines[(lines[:, 1] - lines[:, 0]) >= region_thres]

    return lines


def detect_continuous_areas(image, x_tol=20, y_tol=20, region_thres=0, binary_thres=200):
    """detect rectangles which have continuous pixels

    Args:
        image: 3-D array(h, w, c) or 2-D array(h, w)
        x_tol: see also `detect_continuous_lines()`
        y_tol: see also `detect_continuous_lines()`
        region_thres: see also `detect_continuous_lines()`
        binary_thres: see also `detect_continuous_lines()`

    Returns:

    """
    if len(image.shape) == 3:
        # binary
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, binary_thres, 255, cv2.THRESH_BINARY_INV)

    y_lines = detect_continuous_lines(image, y_tol, region_thres, binary_thres, axis=1)

    bboxes = []

    for y_line in y_lines:
        x_lines = detect_continuous_lines(image[y_line[0]: y_line[1]], x_tol, region_thres, binary_thres, axis=0)
        bbox = np.zeros((len(x_lines), 4), dtype=int)
        bbox[:, 0::2] = x_lines
        bbox[:, 1::2] = y_line
        bboxes.append(bbox)

    bboxes = np.concatenate(bboxes, axis=0)

    return bboxes


class MaskBox:
    """Some definition:
    mask: grey image
        2-d array with shape of (h, w), falls in [0, 255]
    masks: grey images with multi classes
        3-d array with shape of (c, h, w), falls in [0, 255], c gives the classes
    label_mask: label image, each pixel is a classes
        2-d array with shape of (h, w), falls in [0, +inf)
    bbox: a rectangle bounding box of objection
        1-d array with shape of (4, ), 4 gives (x1, y1, x2, y2)
    bboxes: rectangle bounding boxes of objections
        2-d array with shape of (n, 4), 4 gives (x1, y1, x2, y2)
    points: a polygon bounding box of objection
        2-d array with shape of (m, 2), 2 gives (x1, y1)
    segmentations: polygon bounding boxes of objections
        3-d array with shape of (n, m, 2), 2 gives (x1, y1)
    """

    @staticmethod
    def close(image, k_size=8):
        k = np.ones((k_size, k_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, k)

    @staticmethod
    def open(image, k_size=8):
        k = np.ones((k_size, k_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, k)

    @staticmethod
    def label_mask_to_bboxes(label_mask, ignore_class=(), min_area=400, convert_func=None):
        """generate detection bboxes from label mask

        Args:
            label_mask:
            min_area:
            ignore_class (list): usually background class
            convert_func: function to convert the mask

        Returns:

        """
        unique_classes = np.unique(label_mask)
        bboxes = []
        classes = []

        for c in unique_classes:
            if c in ignore_class:
                continue

            mask = (label_mask == c).astype(np.uint8)
            if convert_func:
                mask = convert_func(mask)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_16U)
            stats = stats[stats[:, 4] > min_area]

            stats[:, 2:4] = stats[:, :2] + stats[:, 2:4]
            box = stats[1:, :4]
            bboxes.append(box)
            classes.append([c] * len(box))

        bboxes = np.concatenate(bboxes, axis=0)
        classes = np.concatenate(classes, axis=0)

        return bboxes, classes

    @staticmethod
    def bboxes_to_label_mask(image, bboxes, classes, add_edge=False, edge_cls=255):
        """generate mask from image with detection bboxes"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=image.dtype)

        for box, cls in zip(bboxes, classes):
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = cls

        if add_edge:
            for box, cls in zip(bboxes, classes):
                x1, y1, x2, y2 = box
                mask[y1:y2, x1 - 1 if x1 > 0 else x1] = edge_cls
                mask[y1:y2, x2 + 1 if x2 < w else x2] = edge_cls
                mask[y1 - 1 if y1 > 0 else y1, x1:x2] = edge_cls
                mask[y2 + 1 if y2 < h else y2, x1:x2] = edge_cls

        return mask

    @staticmethod
    def mask_to_bboxes(mask, min_area=0, convert_func=None):
        """generate detection bboxes from mask"""
        if convert_func:
            mask = convert_func(mask)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_16U)
        stats = stats[stats[:, 4] > min_area]

        stats[:, 2:4] = stats[:, :2] + stats[:, 2:4]
        bboxes = stats[1:, :4]
        return bboxes

    @staticmethod
    def mask_to_segmentations(mask, min_area=0, unclip_ratio=0.0, convert_func=None, fix_to_4_edges=True):
        from shapely.geometry import Polygon
        import pyclipper

        if convert_func:
            mask = convert_func(mask)

        def get_mini_boxes(contour):
            bounding_box = cv2.minAreaRect(contour)
            segmentations = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

            index_1, index_2, index_3, index_4 = 0, 1, 2, 3
            if segmentations[1][1] > segmentations[0][1]:
                index_1 = 0
                index_4 = 1
            else:
                index_1 = 1
                index_4 = 0
            if segmentations[3][1] > segmentations[2][1]:
                index_2 = 2
                index_3 = 3
            else:
                index_2 = 3
                index_3 = 2

            new_segmentations = np.array([
                segmentations[index_1], segmentations[index_2], segmentations[index_3], segmentations[index_4]
            ])
            return new_segmentations, min(bounding_box[1])

        def unclip(points):
            poly = Polygon(points)
            distance = poly.area * unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = np.array(offset.Execute(distance))
            return expanded

        outs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[0]
        points_list = []
        for contour in contours:
            points, area = get_mini_boxes(contour)
            if area < min_area:
                continue

            points = unclip(points).reshape(-1, 1, 2)
            points, area = get_mini_boxes(points)
            if area < min_area + 2:
                continue
            points_list.append(points)
        points_list = np.stack(points_list)     # (n, 4, 2)
        return points_list

    @staticmethod
    def segmentations_to_mask(image, segmentations):
        pass


def fragment_image(image: np.ndarray,
                   size=None, grid=None,
                   overlap_size=None, overlap_ratio=None,
                   ignore_size=None, ignore_ratio=None
                   ):
    """fragment a large image to small pieces images
    see also `PIL.Image.split_grid()`

    Args:
        image:
        size:
            e.g. `(100,200)` means tear image with 100*200 pixels
        grid:
            e.g. `(6,7)` means tear image into 6*7 pieces
            note, would be something wrong when overlap_size or overlap_ratio is not None
        overlap_size:
            e.g. `(40, 50)` means one piece has 40 pixels in width overlapping with other pieces
            and 50 pixels in height overlapping with other pieces
        overlap_ratio:
            e.g. `(0.4, 0.5)` means one piece has (0.4 * size) pixels in width overlapping with other pieces
            and (0.5 * size) pixels in height overlapping with other pieces
        ignore_size:
            the size of last image is less than ignore_size will not be fragmented
        ignore_ratio:
            the size of last image is less than (ignore_ratio * size) will not be fragmented

    Usage:

        >>> image = np.zeros((2000, 3000, 3))
        >>> images, coors = fragment_image(image, size=1000, overlap_size=100, ignore_ratio=0.5)
        >>> coors
        [(0, 0, 1000, 1000), (0, 900, 1000, 2000), (900, 0, 1900, 1000), (900, 900, 1900, 2000), (1800, 0, 3000, 1000), (1800, 900, 3000, 2000)]
        >>> [img.shape for img in images]
        [(1000, 1000, 3), (1100, 1000, 3), (1000, 1000, 3), (1100, 1000, 3), (1000, 1200, 3), (1100, 1200, 3)]
    """
    h, w = image.shape[:2]

    if size:
        size = (size, size) if isinstance(size, int) else size
    elif grid:
        size = (int(np.ceil(h / grid)), int(np.ceil(w / grid)))
    else:
        raise f'must be set max_size or grid, can not be None all of them'

    if overlap_size:
        overlap_size = (overlap_size, overlap_size) if isinstance(overlap_size, int) else overlap_size
    elif overlap_ratio:
        overlap_ratio = (overlap_ratio, overlap_ratio) if isinstance(overlap_ratio, float) else overlap_ratio
        overlap_size = (int(size[0] * overlap_ratio[0]), int(size[1] * overlap_ratio[1]))
    else:
        overlap_size = (0, 0)

    if ignore_size:
        ignore_size = (ignore_size, ignore_size) if isinstance(ignore_size, int) else ignore_size
    elif ignore_ratio:
        ignore_ratio = (ignore_ratio, ignore_ratio) if isinstance(ignore_ratio, float) else ignore_ratio
        ignore_size = (int(size[0] * ignore_ratio[0]), int(size[1] * ignore_ratio[1]))
    else:
        ignore_size = (0, 0)

    images = []
    coors = []
    if size:
        for j in range(0, h, size[1] - overlap_size[1]):
            for i in range(0, w, size[0] - overlap_size[0]):
                x1, y1, x2, y2 = i, j, min(i + size[0], w), min(j + size[1], h)

                if w - x1 < ignore_size[0] or h - y1 < ignore_size[1]:
                    continue

                remain = (w - x2, h - y2)
                if remain[0] < ignore_size[0]:
                    x2 = w
                if remain[1] < ignore_size[1]:
                    y2 = h

                coors.append((x1, y1, x2, y2))
                images.append(image[y1:y2, x1:x2])

    return images, coors


def splice_image(images, grid=None, overlap_size=None, pad_values=None):
    """Splicing small pieces images into a large image
    see also `PIL.Image.combine_grid()`

    """

    n = len(images)

    if not n:
        return np.empty((0, 0, 3))

    if grid:
        n_col, n_row = grid
    else:
        if n < 4:
            n_col, n_row = n, 1
        else:  # reshape to square possibly
            n_col = int(np.ceil(np.sqrt(n)))
            n_row = int(np.ceil(n / n_col))

    if overlap_size:
        overlap_size = (overlap_size, overlap_size) if isinstance(overlap_size, int) else overlap_size
        lr, td = overlap_size
        l, t = lr // 2, td // 2
        r, d = lr - l, td - t
    else:
        l, t, r, d = 0, 0, 0, 0

    # Fill to form a complete (n_col * n_row) rectangle
    pad_values = pad_values if pad_values is not None else 0
    pad_image = np.full_like(images[-1], pad_values)
    images += [pad_image] * (n_col * n_row - n)

    rows = []
    for i in range(n_row):
        cols = []
        for j in range(n_col):
            image = images[i * n_col + j]
            h, w = image.shape[:2]

            if j == 0:
                image = image[:, :w - r]
            elif j == n_col - 1:
                image = image[:, l:]
            else:
                image = image[:, l:w - r]

            if i == 0:
                image = image[:h - d]
            elif i == n_row - 1:
                image = image[t:]
            else:
                image = image[t:h - d]

            cols.append(image)

        rows.append(np.concatenate(cols, 1))
    image = np.concatenate(rows, 0)
    return image


def non_max_suppression(boxes, conf, iou_method, threshold=0.6):
    """

    Args:
        boxes (np.ndarray): (n_samples， 4), 4 gives x1,y1,x2,y2
        conf (np.ndarray): (n_samples, )
        iou_method (Callable):
        threshold (float): IOU threshold

    Returns:
        keep (np.ndarray): 1-dim array, index of detections to keep
    """
    index = conf.argsort()[::-1]
    keep = []

    while index.size > 0:
        i = index[0]
        keep.append(i)

        ious = iou_method(boxes[i:i + 1], boxes[index[1:]])[0]
        inds = np.where(ious <= threshold)[0]
        index = index[inds + 1]

    return keep


def lines_to_bboxes(lines, oblique=True):
    """bboxes created by given lines

    Args:
        lines: 2-d array with shape of (n, 4), 4 gives (x1, y1, x2, y2)
        oblique:
            False, all lines perpendicular to axis
            True, any directional lines
    Returns:

    """
    from metrics.object_detection import Overlap

    def cur(points, cur_points=[], dep=0):
        if dep == 3:
            for p in points:
                if p == cur_points[0]:
                    a = set(cur_points)
                    b = {(cur_points[0], cur_points[1]),
                         (cur_points[1], cur_points[2]),
                         (cur_points[2], cur_points[3]),
                         (cur_points[3], cur_points[0])}
                    if a not in all_points:
                        all_points.append(a)
                        all_edges.append(b)
                    break

        for p in points:
            if p in cur_points:
                continue

            cur(dic[p], cur_points + [p], dep + 1)

    obj = Overlap.line2D(lines, lines, return_insert_point=oblique)
    if oblique:
        flag, p = obj
        idx = np.where(flag)
    else:
        flag = obj
        idx = np.where(flag)
        la = lines[idx[0]]
        lb = lines[idx[1]]
        p = np.zeros((*flag.shape, 2))
        p[idx] = np.where(la[:, 0] == la[:, 2], (la[:, 0], lb[:, 1]), (lb[:, 0], la[:, 1])).T

    dic = {}
    for i, j in zip(*idx):
        dic.setdefault(i, []).append(j)

    all_points = []
    all_edges = []

    for k, v in dic.items():
        cur(dic[k], [k])

    all_points = [list(p) for p in all_points]
    all_edges = [list(zip(*edges)) for edges in all_edges]
    rect = []
    for edges in all_edges:
        rect.append(p[edges])

    rect = np.stack(rect)
    bboxes = CoordinateConvert.rect2box(rect)

    return bboxes


class GridBox:
    """Some definition:
    grids: divide an image into nx*ny pieces
        always with property of (edges(l, r, t, d), n_grids(n_grid_x, n_grid_y))
    lines: lines of grids
        2-d array with shape of ((nx+1)*(ny+1), 4), 4 gives (x1, y1, x2, y2)
    cols: cols of grid lines, must be vertical
        1-d array with shape of (nx+1, ), gives the y-axis
    rows: rows of grid lines, must be horizontal
        1-d array with shape of (ny+1, ), gives the x-axis
    points: intersection points of grids lines
        2-d array with shape of ((nx+1)*(ny+1), 2), 2 gives (x, y)
    cells: bounding boxes of grids
        2-d array with shape of (nx*ny, 4), 4 gives (x1, y1, x2, y2)
    bboxes: bounding boxes of objections
        2-d array with shape of (m, 4), 4 gives (x1, y1, x2, y2)
    """

    @staticmethod
    def grids_to_lines(edges, n_grids):
        l, r, t, d = edges
        n_grid_x, n_grid_y = n_grids
        cols = np.linspace(l, r, n_grid_x + 1)
        rows = np.linspace(t, d, n_grid_y + 1)
        return cols, rows

    @classmethod
    def grids_to_points(cls, edges, n_grids):
        n_grid_x, n_grid_y = n_grids

        cols, rows = cls.grids_to_lines(edges, n_grids)

        points_x = np.tile(cols[None, :], (n_grid_x + 1, 1))
        points_y = np.tile(rows[:, None], (1, n_grid_y + 1))

        points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)

        return points

    @classmethod
    def grids_to_cells(cls, edges, n_grids):
        cols, rows = cls.grids_to_lines(edges, n_grids)
        cells = cls.lines_to_cells(cols, rows)
        return cells

    @staticmethod
    def lines_to_points():
        raise NotImplemented

    @staticmethod
    def points_to_lines():
        raise NotImplemented

    @staticmethod
    def points_to_bbox(points):
        crop_width = int(max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])
        ))
        crop_height = int(max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])
        ))
        x1 = min(points[:, 0])
        y1 = min(points[:, 1])
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        bbox = np.array([x1, y1, x2, y2])
        return bbox

    @staticmethod
    def lines_to_cells(cols, rows):
        """cells created by given grid lines"""
        cols = np.sort(cols)
        col1 = cols[:-1]
        col2 = cols[1:]

        rows = np.sort(rows)
        row1 = rows[:-1]
        row2 = rows[1:]

        grid1 = np.meshgrid(col1, row1)
        grid2 = np.meshgrid(col2, row2)

        grid = np.stack(grid1 + grid2)
        grid = np.transpose(grid, (1, 2, 0))
        cells = np.reshape(grid, (-1, 4))
        return cells

    @staticmethod
    def bboxes_inside_cells(bboxes, cells):
        """distinguish the bboxes belonged to(completely included in) the given grid cells

        Returns:
            arg: 1-D array(m, ), m for the index of bboxes, the value for the index of cells
        """
        from metrics.object_detection import Iou
        iou = Iou().u_iou(cells, bboxes)
        arg = np.argmax(iou, axis=1)
        return arg

    @classmethod
    def bboxes_inside_lines(cls, bboxes, cols, rows):
        """distinguish the bboxes belonged to(completely included in) the given grid lines
        """
        cells = cls.lines_to_cells(cols, rows)
        return cls.bboxes_inside_cells(bboxes, cells)

    @staticmethod
    def bboxes_include_cells(bboxes, cells, iou_thres=0.1):
        """distinguish the bboxes included in given grid cells

        Args:
            bboxes:
            cells:
            iou_thres:

        Returns:
            arg: 2-D array(m, n),m for index of bboxes, n for the index of cells
        """
        from metrics.object_detection import Iou
        iou = Iou().u_iou(cells, bboxes)
        arg = iou > iou_thres
        return arg

    @classmethod
    def bboxes_include_lines(cls, bboxes, cols, rows):
        """distinguish the bboxes included in given grid lines
        """
        cells = cls.lines_to_cells(cols, rows)
        return cls.bboxes_include_cells(bboxes, cells)


class ImageCrop:
    @staticmethod
    def bbox_to_rectangle(image, bbox):
        x1, y1, x2, y2 = bbox
        image = image[y1:y2, x1:x2]
        return image

    @staticmethod
    def points_to_polygon(image, points):
        points = points.astype(np.float32)
        crop_width = int(max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])
        ))
        crop_height = int(max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])
        ))
        pts_std = np.float32([
            [0, 0],
            [crop_width, 0],
            [crop_width, crop_height],
            [0, crop_height]
        ])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            image,
            M, (crop_width, crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )
        return dst_img
