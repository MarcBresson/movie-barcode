import numpy as np
import math
import colorsys
from PIL import Image
from collections.abc import Callable

class BarCode:
    METHODS = [
        'defaut',
        'hue',
        'step',
        'intensity',
        'std',
    ]

    def _step(self, pixel, repetitions = 9):
        r,g,b = pixel
        lum = math.sqrt( .241 * r + .691 * g + .068 * b )
        h, s, v = colorsys.rgb_to_hsv(r,g,b)
        h2 = int(h * repetitions)
        v2 = int(v * repetitions)
        if h2 % 2 == 1:
            v2 = repetitions - v2
            lum = repetitions - lum
        return (h2, lum, v2)

    def _hue(self, pixel):
        hue, _, _ = colorsys.rgb_to_hsv(*pixel)
        return hue

    def _saturation(self, pixel):
        _, saturation, _ = colorsys.rgb_to_hsv(*pixel)
        return saturation

    def sort_image(
        self,
        data: np.ndarray,
        method: str = None,
        sorting_function = Callable[[[int, int, int]], int]
    ) -> Image:

        if method == 'hue':
            sorting = np.apply_along_axis(self._hue, 2, data)
        elif method == 'custom':
            sorting = np.apply_along_axis(sorting_function, 2, data)
        elif method == 'intensity':
            sorting = data.sum(2)
        elif method == 'std':
            # similar to saturation, but quite better
            sorting = data.std(2)

        if method == 'defaut':
            img_sorted = data
        elif method == 'step':
            # It does not work when sorting with numpy, because it is sorted *by* a tuple.
            for index in range(len(data)):
                colors_sorted: np.ndarray = data[index]
                colors_sorted = colors_sorted.tolist()

                colors_sorted.sort(key = lambda pixel: self._step(pixel, 8))

                data[index] = np.array(colors_sorted)

            img_sorted = data
        else:
            order = np.argsort(sorting)
            img_sorted = data[np.arange(np.shape(data)[0])[:,np.newaxis], order]

        code_barre = Image.fromarray(img_sorted).transpose(Image.ROTATE_90)

        return code_barre
