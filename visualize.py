"""utils for visualize something(like image, text, etc.)"""
import inspect
import re
from typing import List

import PIL
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from packaging import version

from .excluded.cmap import cmap, terminal_cmap

cmap_list = list(cmap.keys())
POLYGON = 1
RECTANGLE = 2
LEFT_TOP = 0
RIGHT_TOP = 1
RIGHT_DOWN = 2
LEFT_DOWN = 3
CENTER = 4


def get_color_array(idx=None, name=None):
    if idx is not None:
        name = cmap_list[idx % len(cmap_list)]

    color_array = list(cmap[name]['array'])
    color_array[0], color_array[2] = color_array[2], color_array[0]  # rgb to bgr
    return tuple(color_array)


class ImageVisualize:
    @staticmethod
    def box(img, boxes, visual_type=RECTANGLE, colors=None, line_thickness=None, inplace=False):
        """only bbox
        boxes: polygon: (n, k, 2) or rectangle: (n, 4)
        colors: (n, 3) or (n, 1)
        """
        if not inplace:
            img = img.copy()
        colors = colors or [get_color_array(0)] * len(boxes)
        line_thickness = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        for i in range(len(boxes)):
            if visual_type == POLYGON:
                points = np.array(boxes[i], dtype=int)  # polygon: (-1, -1, 2)
                cv2.polylines(img, [points], isClosed=True, color=colors[i], thickness=line_thickness, lineType=cv2.LINE_AA)

            elif visual_type == RECTANGLE:  # rectangle: (-1, 4)
                xyxy = boxes[i]
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(img, c1, c2, color=colors[i], thickness=line_thickness, lineType=cv2.LINE_AA)

            else:
                raise ValueError

        return img

    @staticmethod
    def text_box(img, text_boxes, texts, scores=None, drop_score=0.5, colors=None, font_path="utils/excluded/simfang.ttf"):
        """bbox + text, text needs the text area
        use PIL.Image instead of opencv for better chinese font support
        text_boxes: (n, k, 2)
        colors: (n, 3) or (n, 1)
        """
        scores = scores if scores is not None else [1] * len(text_boxes)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        h, w = image.height, image.width
        img_left = image.copy()
        img_right = Image.new('RGB', (w, h), (255, 255, 255))

        draw_left = ImageDraw.Draw(img_left)
        draw_right = ImageDraw.Draw(img_right)

        colors = colors or [get_color_array(0)] * len(text_boxes)

        for idx, (box, txt, score) in enumerate(zip(text_boxes, texts, scores)):
            if score < drop_score:
                continue

            color = colors[idx]

            box = [tuple(i) for i in box]
            draw_left.polygon(box, fill=color)
            draw_right.polygon(
                [
                    box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                    box[2][1], box[3][0], box[3][1]
                ],
                outline=color)

            box_height = np.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
            box_width = np.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)

            # draw the text
            if box_height > 2 * box_width:
                font_size = max(int(box_width * 0.9), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                cur_x, cur_y = box[0][0] + 3, box[0][1]
                for c in txt:
                    char_size = font.getsize(c) if version.parse(PIL.__version__) < version.parse("9") else font.getbbox(c)
                    draw_right.text((cur_x, cur_y), c, fill=(0, 0, 0), font=font)
                    cur_y += char_size[1]
            else:
                font_size = max(int(box_height * 0.8), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                draw_right.text((box[0][0], box[0][1]), txt, fill=(0, 0, 0), font=font)

        img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(img_right, (w, 0, w * 2, h))

        draw_img = cv2.cvtColor(np.array(img_show), cv2.COLOR_RGB2BGR)

        return draw_img

    @staticmethod
    def text(img, text_boxes, texts, scores=None, drop_score=0.5, font_path="utils/excluded/simfang.ttf"):
        """only text, need text area
        use PIL.Image instead of opencv for better chinese font support
        text_boxes: (n, 4, 2)
        colors: (n, 3) or (n, 1)
        """
        scores = scores if scores is not None else [1] * len(text_boxes)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        draw_image = ImageDraw.Draw(img)

        for idx, (box, txt, score) in enumerate(zip(text_boxes, texts, scores)):
            if score < drop_score:
                continue

            box = np.array(box)
            if box.size == 0:
                continue

            box_height = np.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
            box_width = np.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)

            # draw the text
            if box_height > 2 * box_width:
                font_size = min(max(int(box_width * 0.9), 10), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                cur_x, cur_y = box[0][0] + 3, box[0][1]

                for c in txt:
                    char_size = font.getsize(c) if version.parse(PIL.__version__) < version.parse("9") else font.getbbox(c)
                    draw_image.text((cur_x, cur_y), c, fill=(0, 0, 0), font=font)
                    cur_y += char_size[1]

            else:
                font_size = min(max(int(box_height * 0.8), 10), 10)
                n_font_per_line = max(int(box_width / font_size), 20)
                _txt = ''
                for i in range(0, len(txt), n_font_per_line):
                    _txt += txt[i:i + n_font_per_line] + '\n'
                txt = _txt[:-1]

                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                draw_image.text((box[0][0], box[0][1]), txt, fill=(0, 0, 0), font=font)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    @classmethod
    def label_box(cls, img, boxes, labels, colors=None, point_type=LEFT_TOP, line_thickness=None, inplace=False):
        """boxes + label text, text belong to the box, do not need text area specially
        note, do not support Chinese
        boxes: (n, 4)
        colors: (n, 3) or (n, 1)
        """
        if not inplace:
            img = img.copy()

        line_thickness = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        colors = colors or [get_color_array(0)] * len(boxes)
        labels = [str(i) for i in labels]

        cls.box(img, boxes, visual_type=RECTANGLE, colors=colors, line_thickness=line_thickness, inplace=True)

        # visual label
        for i in range(len(labels)):
            xyxy = boxes[i]
            cls.label(img, labels[i], point=(int(xyxy[0]), int(xyxy[1])), point_type=point_type, bg_color=colors[i], thickness=line_thickness, inplace=True)

        return img

    @staticmethod
    def label(img, label, point=(0, 0), point_type=LEFT_TOP, bg_color=None, font_color=None, thickness=None, inplace=False):
        """only label text, do not need text area"""
        if not inplace:
            img = img.copy()

        bg_color = bg_color or get_color_array(name='Black')
        font_color = font_color or get_color_array(name='White')
        label = str(label)

        thickness = thickness or round(0.01 * (img.shape[0] + img.shape[1]) / 2)
        thickness = max(thickness, 1)  # font thickness
        font_scale = thickness / 5

        t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=thickness)[0]

        if point_type == LEFT_TOP:
            bg_lt = point

        elif point_type == LEFT_DOWN:
            bg_lt = (point[0], point[1] - t_size[1] - thickness * 2)

        elif point_type == RIGHT_TOP:
            bg_lt = (point[0] - t_size[0], point[1])

        elif point_type == RIGHT_DOWN:
            bg_lt = (point[0] - t_size[0], point[1] - t_size[1] - thickness * 2)

        elif point_type == CENTER:
            bg_lt = (point[0] - t_size[0] // 2, point[1] - (t_size[1] + thickness * 2) // 2)

        else:
            raise ValueError(f'{point_type} not supported')

        bg_rd = (bg_lt[0] + t_size[0], bg_lt[1] + t_size[1] + thickness * 2)
        text_ld = (bg_lt[0], bg_lt[1] + t_size[1] + thickness)
        cv2.rectangle(img, bg_lt, bg_rd, bg_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, text_ld, 0, font_scale, font_color, thickness=thickness, lineType=cv2.LINE_AA)

        return img

    @staticmethod
    def block(img, boxes, visual_type=RECTANGLE, colors=None, alpha=1):
        """目标块
        boxes: polygon: (n, -1, 2) or rectangle: (n, 4)
        alpha: falls in [0, 1], 1 gives opaque totally
        colors: (n, 3) or (n, 1)
        """
        img = img.copy()
        colors = colors or [get_color_array(0)] * len(boxes)

        for i in range(len(boxes)):
            if visual_type == POLYGON:  # polygon: (-1, -1, 2)
                mask = img.copy()
                cv2.fillPoly(mask, [np.array(boxes[i], dtype=int)], color=colors[i], lineType=cv2.LINE_AA)
                img = (img * (1 - alpha) + mask * alpha).astype(img.dtype)

            elif visual_type == RECTANGLE:  # rectangle: (-1, 4)
                boxes = np.array(boxes).astype(int)
                x1, y1, x2, y2 = boxes[i]
                block = img[y1:y2, x1:x2]
                img[y1:y2, x1:x2] = (block * (1 - alpha) + (np.zeros_like(block) + colors[i]) * alpha).astype(img.dtype)

            else:
                raise ValueError

        return img

    @staticmethod
    def mask(img, mask):
        mask = mask.astype(np.float32)[:, :, None]
        mask /= 255
        img = (img * mask).astype(img.dtype)
        return img

    @staticmethod
    def label_mask(img, label_mask, max_class=None, colors=None, alpha=0.5, ignore_class=()):
        if max_class is None:
            max_class = np.max(label_mask)

        if colors is None:
            colors = [get_color_array(i) for i in range(max_class + 1)]

        mask = img.copy()
        for i in range(max_class + 1):
            if i in ignore_class:
                continue
            mask[label_mask == i] = colors[i]

        img = (img * (1 - alpha) + mask * alpha).astype(img.dtype)
        return img


def get_variable_name(var):
    # there may be some bugs in the future
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class TextVisualize:
    @staticmethod
    def get_start(types=None):
        if not types:
            return ''

        types = [types] if isinstance(types, str) else types

        # fmt -> \033[%sm
        start = '\033['
        for t in types:
            start += terminal_cmap[t] + ';'

        start = start[:-1] + 'm'

        return start

    @staticmethod
    def get_end(types=None):
        if not types:
            return ''
        return '\033[' + terminal_cmap['end'] + 'm'

    @classmethod
    def highlight_str(cls, text, types=None, fmt='', start='', end='', return_list=False, **kwargs):
        """hightlight a string

        Args:
            text(str or tuple):
                apply for `fmt % text`
            types(str or tuple):
                one of keys of `terminal_cmap',
                unuse when start and end is set,
                if fmt is not set, default is `('blue', 'bold')`
            fmt(str): highlight format, fmt like '<left>%s<right>'
            start(object):
            end(object)
            return_list(bool)

        Examples:
            >>> TextVisualize.highlight_str('hello')
            >>> TextVisualize.highlight_str('hello', 'blue')
            >>> TextVisualize.highlight_str('hello', ('blue', 'bold'))
            >>> TextVisualize.highlight_str('hello', fmt='<p style="color:red;">%s</p>')    # html type
            >>> TextVisualize.highlight_str('hello', 'blue', fmt='(highlight str: %s)')  # add special text

        """
        if not (types or fmt):
            types = ('blue', 'bold')

        if types:
            start = start or cls.get_start(types)
            end = end or cls.get_end(types)

        if not fmt:
            fmt = '%s'

        fmt = fmt % text
        s = [start, fmt, end]
        return s if return_list else ''.join(s)

    @classmethod
    def highlight_subtext(cls, text, span, highlight_obj=None,
                          keep_len=None, left_abbr='...', right_abbr='...',
                          auto_truncate=False, truncate_pattern=None,
                          return_list=False, **kwargs):
        """highlight a string where giving by span of a text
        See Also `TextVisualize.highlight_str`

        Args:
            text(str):
            span(tuple):
            highlight_obj(List[str]):
            keep_len(int):
                limit output str length, the len gives the length of left and right str.
                No limit if None, or exceeding part collapse to abbr str
            left_abbr(str)
            right_abbr(str)
            auto_truncate(bool): if true, truncate the text
            truncate_pattern(re.Patern)
            return_list(bool)
            kwargs: see also `TextVisualize.highlight_str()` to get more info

        Examples:
            >>> TextVisualize.highlight_subtext('123,4567,890abc', (5, 7))
            >>> TextVisualize.highlight_subtext('123,4567,890abc', (5, 7), keep_len=5)
            >>> TextVisualize.highlight_subtext('123,4567,890abc', (5, 7), keep_len=5, auto_truncate=True)

        """
        if not highlight_obj:
            highlight_obj = cls.highlight_str(text[span[0]:span[1]], return_list=return_list, **kwargs)
        highlight_obj = highlight_obj if return_list else [highlight_obj]

        if keep_len:
            left = max(0, span[0] - keep_len)
            right = min(len(text), span[1] + keep_len)

            if auto_truncate:
                truncate_pattern = truncate_pattern or re.compile(r'[。\.!\?！？;；,，]')
                a = text[left:right]
                r = truncate_pattern.split(a)
                if len(r) >= 3:  # make sure that returning one sentence at least
                    if left > 0 and not truncate_pattern.match(text[left - 1]):
                        _ = left + len(r[0]) + 1
                        if _ < span[0]:
                            left = _

                    if right < len(text) - 1 and not truncate_pattern.match(text[right + 1]):
                        _ = right - len(r[-1])
                        if _ > span[1]:
                            right = _

            left_abbr = left_abbr if left > 0 else ''
            right_abbr = right_abbr if right < len(text) else ''

            s = [
                left_abbr,
                text[left:span[0]],
                *highlight_obj,
                text[span[1]:right],
                right_abbr,
            ]

        else:
            s = [
                text[:span[0]],
                *highlight_obj,
                text[span[1]:]
            ]

        return s if return_list else ''.join(s)

    @classmethod
    def highlight_subtexts(cls, text, spans, highlight_objs=None, fmt='', return_list=False, ignore_overlap=False, **kwargs):
        """highlight multiple strings where giving by spans of a text
        See Also `TextVisualize.highlight_str`

        Args:
            text(str):
            spans(List[tuple]):
            highlight_objs(List[List[str]):
            fmt(str or list):
            return_list(bool):
            ignore_overlap(bool):
            kwargs: see also `TextVisualize.highlight_str()` to get more info

        Examples:
            >>> TextVisualize.highlight_subtexts('hello world', [(2, 3), (6, 7)])

        """
        if not spans:
            return text

        arg = np.argsort(spans, axis=0)

        s = []
        a = 0
        for i in arg[:, 0]:
            span = spans[i]
            if a > span[0]:
                if ignore_overlap:
                    print(f'{span = } overlap, please check')
                    continue
                else:
                    raise f'{span = } overlap, please check'

            if highlight_objs:
                highlight_obj = highlight_objs[i]
            else:
                _fmt = fmt[i] if isinstance(fmt, list) else fmt
                highlight_obj = cls.highlight_str(text[span[0]:span[1]], fmt=_fmt, return_list=return_list, **kwargs)

            highlight_obj = highlight_obj if return_list else [highlight_obj]
            s += [text[a:span[0]]] + highlight_obj
            a = span[1]

        s.append(text[a:])

        return s if return_list else ''.join(s)

    @classmethod
    def mark_subtext(cls, text, span, mark, types=('blue', 'bold'), fmt='', **kwargs):
        """highlight a string with mark symbols

        Examples:
            >>> TextVisualize.mark_subtext('hello', (2, 4), 'ii')
            >>> TextVisualize.mark_subtext('hello', (2, 4), 'ii', fmt='%s(to %s)')
        """
        fmt = fmt or '(%s -> %s)'
        highlight_obj = cls.highlight_str((text[span[0]:span[1]], mark), types=types, fmt=fmt, **kwargs)
        return cls.highlight_subtext(text, span, highlight_obj, **kwargs)

    @classmethod
    def mark_subtexts(cls, text, spans, marks, types=('blue', 'bold'), fmt='', **kwargs):
        """
        Examples:
            >>> TextVisualize.mark_subtexts('hello world', [(2, 3), (6, 7)], ['i', 'v'])
            >>> TextVisualize.mark_subtexts('hello world', [(2, 3), (6, 7)], ['i', 'v'], fmt='%s(to %s)')
        """
        _fmt = []
        highlight_objs = []
        for i, (span, mark) in enumerate(zip(spans, marks)):
            _fmt = fmt[i] if isinstance(fmt, list) else fmt
            _fmt = _fmt or '(%s -> %s)'
            highlight_obj = cls.highlight_str((text[span[0]:span[1]], mark), types=types, fmt=_fmt, **kwargs)
            highlight_objs.append(highlight_obj)
        return cls.highlight_subtexts(text, spans, highlight_objs, **kwargs)

    @staticmethod
    def num_to_human_readable_str(num: int, factor: float | list = 1024., suffixes=('b', 'K', 'M', 'G', 'T')):
        """
        Examples:
            >>> TextVisualize.num_to_human_readable_str(1234567)
            1.18 M
            >>> TextVisualize.num_to_human_readable_str(1234567, factor=1e3)
            1.23 M
            >>> TextVisualize.num_to_human_readable_str(1234567, factor=(60., 60., 24.), suffixes=('s', 'm', 'h'))
            14.29 h
        """
        if not isinstance(factor, (list, tuple)):
            factor = [factor] * len(suffixes)

        for suffix, f in zip(suffixes, factor):
            if num >= f:
                num /= f
            else:
                return f'{num:.2f} {suffix}'

        return f'{num:.2f} {suffix}'

    @classmethod
    def dict_to_str(cls, dic: dict, return_list=False):
        """
        Examples:
            >>> TextVisualize.dict_to_str({'a': 1, 'b': {'c': 2, 'd': 3}})
            a=1,b.c=2,b.d=3
        """
        s = []
        for k, v in dic.items():
            if isinstance(v, dict):
                v = cls.dict_to_str(v, return_list=True)
                for vv in v:
                    s.append(f'{k}.{vv}')
            else:
                s.append(f'{k}={v}')
        return s if return_list else ','.join(s)


NORMAL = 0
POLAR = 1


class Formulae2DVisualize:
    """Visualize by `matplotlib`
    Some definition:
        implicit formulae:
            e.g. `f(x, y) = 0`
        explicit formulae:
            e.g. `y = f(x)`
        transform formulae:
            e.g. `x = f1(t), y = f2(t)`
        differential formulae:
            e.g. `dx = f_x(x, y)dt, dy = f_y(x, y)dt`
    """

    @staticmethod
    def _vis(
            ax, fig,
            xlim=None, ylim=None, axis=False,
            title='', facecolor='papayawhip',
            save_path=None, show_fig=True,
            **kwargs
    ):
        import matplotlib.pyplot as plt

        xlim = xlim or ax.get_xlim()
        ylim = ylim or ax.get_ylim()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title)

        if not axis:
            ax.set_axis_off()

        fig.set_facecolor(facecolor)
        ax.set_facecolor(facecolor)

        if save_path is not None:
            plt.savefig(save_path, facecolor=fig.get_facecolor())

        if show_fig:
            plt.show()

    @classmethod
    def implicit_by_matplotlib(
            cls,
            func,
            steps=2 ** 10,
            xaxis=None, yaxis=None,
            taxis=None, raxis=None,
            axis_type=NORMAL,
            **draw_kwargs
    ):
        """plot by method of `contour` module from matplotlib

        Args:
            func:
            steps:
            xaxis:
            yaxis:
            taxis:
            raxis:
            axis_type:
            **draw_kwargs:

        Returns:

        Examples:
            from numpy import exp, sin, cos

            Formulae2DVisualize.implicit_by_matplotlib(
                func=lambda x, y: exp(sin(x) + cos(y)) - sin(exp(x + y)),
                xaxis=(-10, 10), yaxis=(-10, 10),
                title=r'$e^{\sin (x) + \cos (y)} = \sin (e^{x + y})$',
            )

        """
        from numpy import sin, cos
        import matplotlib.pyplot as plt

        if axis_type == POLAR:
            xaxis, yaxis = taxis, raxis

        x = np.linspace(*xaxis, steps)
        y = np.linspace(*yaxis, steps)
        xx, yy = np.meshgrid(x, y)
        zz = func(xx, yy)

        if axis_type == POLAR:
            xx, yy = yy * cos(xx), yy * sin(xx)

        fig = plt.figure(figsize=draw_kwargs.get('figsize', None))
        ax = fig.add_subplot()
        ax.contour(xx, yy, zz, 0, colors=draw_kwargs.get('color', 'r'))

        cls._vis(ax, fig, **draw_kwargs)

        return ax, (xx, yy, zz)

    @classmethod
    def implicit_by_sympy(
            cls,
            func,
            steps=2 ** 10,
            xaxis=None, yaxis=None,
            taxis=None, raxis=None,
            axis_type=NORMAL,
            **draw_kwargs
    ):
        """plot by method of `plot_implicit` module from sympy

        Args:
            func:
            steps:
            xaxis:
            yaxis:
            taxis:
            raxis:
            axis_type:
            **draw_kwargs:

        Returns:

        Examples:
            from sympy import exp, sin, cos

            Formulae2DVisualize.implicit_by_sympy(
                func=lambda x, y: exp(sin(x) + cos(y)) - sin(exp(x + y)),
                xaxis=(-10, 10), yaxis=(-10, 10),
                title=r'$e^{\sin (x) + \cos (y)} = \sin (e^{x + y})$',
            )
        """
        from sympy import symbols, plot_implicit

        if axis_type == POLAR:
            xaxis, yaxis = taxis, raxis

        x, y = symbols('x y')
        z = func(x, y)

        p = plot_implicit(z, (x, *xaxis), (y, *yaxis),
                          points=steps, line_color=draw_kwargs.get('color', 'r'), show=False,
                          xlim=draw_kwargs.get('xlim', (None, None)), ylim=draw_kwargs.get('ylim', (None, None)),
                          title=draw_kwargs.get('title', ''))

        if draw_kwargs.get('save_path', None) is not None:
            p.save(draw_kwargs.get('save_path'))

        p.show()

    @classmethod
    def implicit_by_gradient(
            cls,
            func,
            steps=2 ** 10,
            xaxis=None, yaxis=None,
            taxis=None, raxis=None,
            axis_type=NORMAL,
            k=1.,
            **draw_kwargs
    ):
        """plot by method of gradient

        Args:
            func:
            steps:
            xaxis:
            yaxis:
            taxis:
            raxis:
            axis_type:
            k:
            **draw_kwargs:

        Returns:

        Examples:
            from numpy import exp, sin, cos

            Formulae2DVisualize.implicit_by_gradient(
                func=lambda x, y: exp(sin(x) + cos(y)) - sin(exp(x + y)),
                xaxis=(-10, 10), yaxis=(-10, 10),
                title=r'$e^{\sin (x) + \cos (y)} = \sin (e^{x + y})$',
            )
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        if axis_type == POLAR:
            xaxis, yaxis = taxis, raxis

        x = np.linspace(*xaxis, steps)
        y = np.linspace(*yaxis, steps)
        xx, yy = np.meshgrid(x, y)
        zz = func(xx, yy)

        grad = np.gradient(zz)
        mold = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
        zz = np.abs(zz)

        a = zz < k * mold
        a = a[::-1, :]

        fig = plt.figure(figsize=draw_kwargs.get('figsize', None))
        ax = fig.add_subplot()

        cmap = colors.ListedColormap([draw_kwargs.get('facecolor', 'papayawhip'), 'red'])
        ax.imshow(a, cmap=cmap)

        cls._vis(ax, fig, **draw_kwargs)
        return ax, (xx, yy, zz)

    @classmethod
    def implicit_by_ms(
            cls,
            func,
            steps=2 ** 10,
            xaxis=None, yaxis=None,
            taxis=None, raxis=None,
            axis_type=NORMAL,
            **draw_kwargs
    ):
        """plot by method of Marching squares algorithm

        Args:
            func:
            steps:
            xaxis:
            yaxis:
            taxis:
            raxis:
            axis_type:
            **draw_kwargs:

        Returns:

        Examples:
            from numpy import exp, sin, cos

            Formulae2DVisualize.implicit_by_ms(
                func=lambda x, y: exp(sin(x) + cos(y)) - sin(exp(x + y)),
                xaxis=(-10, 10), yaxis=(-10, 10),
                title=r'$e^{\sin (x) + \cos (y)} = \sin (e^{x + y})$',
            )
        """
        from numpy import sin, cos
        import matplotlib.pyplot as plt
        import matplotlib.collections as collections

        if axis_type == POLAR:
            xaxis, yaxis = taxis, raxis

        x = np.linspace(*xaxis, steps)
        y = np.linspace(*yaxis, steps)
        xx, yy = np.meshgrid(x, y)
        zz = func(xx, yy)

        if axis_type == POLAR:
            x, y = y * cos(x), y * sin(x)

        r = np.zeros_like(zz, dtype=int)
        r[zz > 0] = 1

        mask = np.zeros((4, r.shape[0], r.shape[1]), dtype=int)
        mask[0], mask[1], mask[2], mask[3] = 2, 3, 4, 5
        mask = mask * r

        w = np.zeros((4, r.shape[0] - 1, r.shape[1] - 1), dtype=int)
        w[0], w[1], w[2], w[3] = mask[0][:-1, :-1], mask[1][:-1, 1:], mask[2][1:, :-1], mask[3][:-1, :-1]
        w[w == 0] = 1

        sites = np.prod(w, axis=0)

        middle_x, middle_y = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2

        site_map = {
            2: [[0, 3]],
            3: [[0, 1]],
            4: [[2, 3]],
            5: [[1, 2]],
            6: [[1, 3]],
            8: [[0, 2]],
            10: [[0, 1], [2, 3]],
            12: [[0, 3], [1, 2]],
            15: [[0, 2]],
            20: [[1, 3]],
            24: [[1, 2]],
            30: [[2, 3]],
            40: [[0, 1]],
            60: [[0, 3]],
        }

        lines = []

        for i in range(r.shape[1] - 1):
            for j in range(r.shape[0] - 1):
                site = sites[j, i]

                if site == 1 or site == 120:
                    continue

                point_map = [(middle_x[i], y[j]),
                             (x[i + 1], middle_y[j]),
                             (middle_x[i], y[j + 1]),
                             (x[i], middle_y[j])]

                points = site_map[site]

                for point in points:
                    lines.append([point_map[point[0]], point_map[point[1]]])

        fig = plt.figure(figsize=draw_kwargs.get('figsize', None))
        ax = fig.add_subplot()

        lc = collections.LineCollection(lines, colors=draw_kwargs.get('color', 'r'))
        ax.add_collection(lc, autolim=True)
        ax.autoscale_view()

        cls._vis(ax, fig, **draw_kwargs)
        return ax, (xx, yy, zz)

    @classmethod
    def implicit_by_ti(
            cls,
            func,
            steps=2 ** 10,
            xaxis=None, yaxis=None,
            taxis=None, raxis=None,
            axis_type=NORMAL,
            **draw_kwargs
    ):
        """plot by method of Tupper interval arithmetic"""
        raise NotImplemented

    @classmethod
    def explicit(
            cls,
            func,
            steps: int = 2 ** 10,
            axis_type: int = NORMAL,
            axis_list: iter = None,
            **draw_kwargs
    ):
        """

        Args:
            func:
            steps:
            axis_type:
            axis_list:
            **draw_kwargs:

        Returns:

        Examples:
            from numpy import sin, cos, pi

            # visual a normal formulae
            Formulae2DVisualize.explicit(
                func=lambda x: abs(x) ** (2 / 3) + 0.9 * (3.3 - x ** 2) ** 0.5 * sin(30 * pi * x),
                axis_list=((-3.3 ** 0.5, 3.3 ** 0.5),),
                steps=2 ** 10,
                title=r'$y = \sqrt[3]{x^2} + 0.9(3.3 - x^2)^{0.5} \cdot  \frac{\sin (b \pi  x)}{nb} $' + '\n' +
                      r'$n=30$',
                save_path='test.png',
            )

            # visual a polar formulae
            Formulae2DVisualize.explicit(
                func=lambda t: 1 - sin(t),
                axis_type=POLAR,
                axis_list=((-pi * 2, pi * 2),),
                steps=2 ** 10,
                title=r'$r = 1 - \sin (t)$',
                save_path='test.png',
            )

        """
        from numpy import sin, cos
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=draw_kwargs.get('figsize', None))
        ax = fig.add_subplot()

        xs, ys = [], []
        for axis in axis_list:
            x = np.linspace(*axis, steps)
            y = func(x)
            if axis_type == POLAR:
                x, y = y * cos(x), y * sin(x)

            xs.append(x)
            ys.append(y)

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)

        ax.plot(xs, ys, c=draw_kwargs.get('color', 'r'), linewidth=draw_kwargs.get('linewidth', 2))
        cls._vis(ax, fig, **draw_kwargs)
        return ax, (xs, ys)

    @classmethod
    def transform(
            cls,
            func,
            steps=2 ** 10,
            taxis_list=None,
            axis_type=NORMAL,
            **draw_kwargs
    ):
        """

        Args:
            steps:
            func:
            taxis_list:
            axis_type:
            **draw_kwargs:

        Returns:

        Examples:
            from numpy import sin, cos, pi, log, sqrt

            Formulae2DVisualize.transform(
                func=lambda t: (sin(t) * cos(t) * log(abs(t)),
                                sqrt(abs(t)) * cos(t)),
                steps=2 ** 10,
                taxis_list=((-1, 1),),
                title=r'$x = \sin (t) \cos (t) \ln (|t|)$' + '\n' +
                      r'$y = \sqrt{|t|} \cos (t)$' + '\n' +
                      r'$n-1 \leq t \leq 1$',
            )
        """
        from numpy import sin, cos
        import matplotlib.pyplot as plt

        xs, ys = [], []
        for taxis in taxis_list:
            t = np.linspace(*taxis, steps)
            x, y = func(t)
            if axis_type == POLAR:
                x, y = y * cos(x), y * sin(x)

            xs.append(x)
            ys.append(y)

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)

        fig = plt.figure(figsize=draw_kwargs.get('figsize', None))
        ax = fig.add_subplot()

        ax.plot(xs, ys, c=draw_kwargs.get('color', 'r'))
        cls._vis(ax, fig, **draw_kwargs)
        return ax, (xs, ys)

    @classmethod
    def differential(
            cls,
            func,
            start_values=(0, 0),
            steps=2 ** 10,
            taxis_list=None,
            **draw_kwargs
    ):
        """

        Args:
            func:
            start_values:
            steps:
            taxis_list:
            **draw_kwargs:

        Returns:

        Examples:
            Formulae2DVisualize.differential(
                func=lambda x, y, dt: (
                    (-3 * y - x ** 2) * dt,
                    (3 ** 0.5 * x - y ** 3) * dt
                ),
                steps=10 ** 5,
                taxis_list=[(0, 100)],
                start_values=(1, 1),
            )

        """
        import matplotlib.pyplot as plt

        total_steps = 0
        ts = []
        for taxis in taxis_list:
            ts.append(np.linspace(*taxis, steps + 1))
            total_steps += steps + 1
        ts = np.concatenate(ts)

        xs = np.zeros(total_steps)
        ys = np.zeros(total_steps)

        xs[0], ys[0] = start_values

        for i in range(total_steps - 1):
            x, y, t = xs[i], ys[i], ts[i]
            dt = ts[i + 1] - t
            dx, dy = func(x, y, dt)

            xs[i + 1] = xs[i] + dx
            ys[i + 1] = ys[i] + dy

        fig = plt.figure(figsize=draw_kwargs.get('figsize', None))
        ax = fig.add_subplot()
        ax.plot(xs, ys, c=draw_kwargs.get('color', 'r'), linewidth=draw_kwargs.get('linewidth', 2))
        cls._vis(ax, fig, **draw_kwargs)
        return ax, (xs, ys)


class Formulae3DVisualize:
    """Visualize by mathplotlib
    Some definition:
        implicit formulae:
            e.g. `f(x, y, z) = 0`
        transform formulae:
            e.g. `x = f1(t), y = f2(t), z = f3(t)`
        differential formulae:
            e.g. `dx = f1(x, y, z)dt, dy = f2(x, y, z)dt`
    """

    @classmethod
    def implicit(cls):
        raise NotImplemented

    @classmethod
    def transform(
            cls,
            func,
            steps=2 ** 10,
            taxis_list=None,
            **draw_kwargs
    ):
        import matplotlib.pyplot as plt

        xs, ys, zs = [], [], []
        for taxis in taxis_list:
            t = np.linspace(*taxis, steps)
            x, y, z = func(t)

            xs.append(x)
            ys.append(y)
            zs.append(z)

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        zs = np.concatenate(zs)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot(xs, ys, zs, linewidth=.5)
        Formulae2DVisualize._vis(ax, fig, **draw_kwargs)
        return ax, (xs, ys, zs)

    @classmethod
    def differential(
            cls,
            func,
            start_values=(0, 0, 0),
            steps=2 ** 10,
            taxis_list=None,
            **draw_kwargs
    ):
        """

        Args:
            func:
            start_values:
            steps:
            taxis_list:
            **draw_kwargs:

        Returns:

        Examples:
            def lorenz(x, y, z, dt, s=10, r=28, b=2.667):
                dx = s * (y - x)
                dy = r * x - y - x * z
                dz = x * y - b * z
                return dx * dt, dy * dt, dz * dt

            Formulae3DVisualize.differential(
                func=lorenz,
                n_steps=10 ** 5,
                start_values=(0.1, 0.1, 0.1),
                taxis_list=[(0, 10)]
            )
        """
        import matplotlib.pyplot as plt

        total_steps = 0
        ts = []
        for taxis in taxis_list:
            ts.append(np.linspace(*taxis, steps + 1))
            total_steps += steps + 1
        ts = np.concatenate(ts)

        xs = np.zeros(total_steps)
        ys = np.zeros(total_steps)
        zs = np.zeros(total_steps)

        xs[0], ys[0], zs[0] = start_values

        for i in range(total_steps - 1):
            x, y, z, t = xs[i], ys[i], zs[i], ts[i]
            dt = ts[i + 1] - t
            dx, dy, dz = func(x, y, z, dt)

            xs[i + 1] = xs[i] + dx
            ys[i + 1] = ys[i] + dy
            zs[i + 1] = zs[i] + dz

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot(xs, ys, zs, linewidth=.5)
        Formulae2DVisualize._vis(ax, fig, **draw_kwargs)
        return ax, (xs, ys, zs)
