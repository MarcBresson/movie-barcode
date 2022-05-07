import numpy as np
import logging
import cv2
import av
import os
import sys
import shutil
from PIL import Image
from typing import Literal
from collections.abc import Callable
from movie_barcode.barcode import BarCode
from movie_barcode.timer import Timer

logger = logging.getLogger(__name__)

class Movie:
    def __init__(
        self,
        path: str
    ) -> None:

        self.input_path = path

        file_name = path.split("/")[-1].split(".")
        file_name.pop()
        self.file_name = "_".join(file_name)

        self.colors = None

        self._count_total_frame()

        self.set_generation_technique()



    def _count_total_frame(self):
        cap = cv2.VideoCapture(self.input_path)
        self.total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()



    def compute_colors(
        self,
        id_video_flux: int = 0
    ) -> np.ndarray:
        """Compute dominants color for every other frame.
        
        Example
        ~~~~~~~~
        movie.compute_colors()

        Parameters
        ~~~~~~~~
        :param int id_video_flux:
            Typically 0, except if your movie has multiple video flux.
        """

        # Load the movie
        movie = av.open(self.input_path)
        movie.streams.video[id_video_flux].thread_type = 'AUTO'

        # Create the empty json
        dominant_colors = np.zeros(
            (self.nbr_frame_to_compute, self.color_palette_size, 3),
            dtype=np.uint8
        )

        sys.stdout.write(self.file_name + ":\n")

        timer = Timer(self.nbr_frame_to_compute)
        frame_computed = 0
        # We must go through every frame due to encoding specificity
        for frame_index, frame in enumerate(movie.decode(video=id_video_flux)):

            if frame_index > (frame_computed * self.total_frames / self.nbr_frame_to_compute) :

                tableau_image = self._get_dominant_colors(frame.to_image(), self.color_palette_size)
                dominant_colors[frame_computed] = tableau_image

                frame_computed += 1
                self._display_progress_bar(frame_computed, self.nbr_frame_to_compute, timer)

        self.colors = dominant_colors

        sys.stdout.write(" " * shutil.get_terminal_size().columns)



    def _display_progress_bar(
        self,
        frame: int,
        total_frame: int,
        timer: Timer,
        ch: str = "█",
        scale: float = 0.55
    ) -> None:
        """Display a simple, pretty progress bar. From pytube.
        
        Example
        ~~~~~~~~
        thanks_pytube_dev.mp4

        ↳ |███████████████████████████████████████| 100.0%

        Parameters
        ~~~~~~~~
        :param int frame:
            The frame which is being computed.
        :param int total_frame:
            Total number of frame of the media loaded.
        :param str ch:
            Character to use for presenting progress segment.
        :param float scale:
            Scale multiplier to reduce progress bar size.
        """

        columns = shutil.get_terminal_size().columns
        max_width = int(columns * scale)

        filled = int(round(max_width * frame / float(total_frame)))
        remaining = max_width - filled
        progress_bar = ch * filled + " " * remaining
        percent = round(100.0 * frame / float(total_frame), 1)
        text = f" ↳ |{progress_bar}| {percent}% - {timer.remains(frame)} remaining \r"
        sys.stdout.write(text)
        sys.stdout.flush()



    def _get_dominant_colors(
        self,
        pil_img: Image,
        color_palette_size: int
    ) -> np.ndarray:

        # Resize image to speed up processing
        img = pil_img.copy()
        img.thumbnail((200, 200),Image.BOX)

        if False:
            img_colonne = pil_img.copy()
            img_colonne.resize((1,1080),Image.BOX)

        # Reduce colors (uses k-means internally)
        paletted = img.convert(
            'P',
            palette=Image.ADAPTIVE,
            colors = color_palette_size
        )

        # Find the color that occurs most often
        palette = paletted.getpalette()
        color_counts = sorted(paletted.getcolors(), reverse=True)

        # Create the ordered palette
        ordered_palette = np.zeros(
            (len(color_counts), 3),
            dtype=np.uint8
        )
        for i in range(len(color_counts)):
            palette_index = color_counts[i][1] * 3
            ordered_palette[i] = palette[palette_index : palette_index+3]

        return np.resize(ordered_palette, (color_palette_size, 3))



    def save(
        self,
        dir_path: str,
    ) -> None:
        """Export the computed npy array to the specified directory.
        
        Example
        ~~~~~~~~
        movie.save("path/to/directory")

        Parameters
        ~~~~~~~~
        :param str dir_path:
            Directory path to where export the npy file.
        """

        if not isinstance(self.colors, np.ndarray):
            logger.error("Colors not computed yet. Can't save the file.")
            return None

        file_path = os.path.join(dir_path, self.file_name + '-' + self.technique)
        np.save(file_path, self.colors)
        logger.info(f"{file_path} : saved succefully !")



    def load(
        self,
        dir_path: str,
        failure_autorized = False
    ) -> None:
        """Import the computed npy array from the specified directory.
        
        Example
        ~~~~~~~~
        movie.load("path/to/directory")

        Parameters
        ~~~~~~~~
        :param str dir_path:
            Directory path from where import the npy file.
        """

        file_path = os.path.join(dir_path, self.file_name)
        try:
            self.colors = np.load(file_path + '-' + self.technique + '.npy')
            logger.info(f"{file_path} : found and loaded.")
        except FileNotFoundError:

            if not failure_autorized:
                logger.info(f"{file_path} : not found.")
                raise FileNotFoundError
            else:
                pass



    def export_barcode(
        self,
        method: str,
        dir_path: str,
        sorting_function = Callable[[[int, int, int]], int]
    ) -> None:
        """Generate and export a barcode.
        
        Example
        ~~~~~~~~
        movie.export_barcode('hue', 'path/to/a/directory')
        movie.export_barcode('step', 'path/to/directory', sorting_function)

        Parameters
        ~~~~~~~~
        :param str method:
            Sorting method to apply to each bar.
        :param str dir_path:
            Directory path to where export the barcode generated.
        :param Callable sorting_function:
            Function for the sorting algorithm. The input is
            a pixel [red, green, blue]. The output is a scalar
            to sort from.
        """

        code_barre: Image = BarCode().sort_image(
            self.colors,
            method,
            sorting_function
        )

        code_barre = code_barre.resize((self.output_width, self.output_height), Image.NEAREST)
        code_barre.save(os.path.join(dir_path, f'{self.file_name} barcode {method}.png'))



    def export_every_barcode(
        self,
        dir_path: str,
    ) -> None:
        """Generate and export barcodes using every sorting function.
        
        Example
        ~~~~~~~~
        movie.export_every_barcode('dir/to/barcodes')

        Parameters
        ~~~~~~~~
        :param str dir_path:
            Directory path to where export the barcode generated.
        """

        for method in BarCode.METHODS:
            self.export_barcode(
                method,
                dir_path
            )



    def set_generation_technique(
        self,
        technique: Literal['dominant', 'squeeze'] = 'dominant',
        output_height: int = 1280,
        output_width: int = 1920 * 2,
        color_palette_size: int = 1280/5,
    ) -> None:

        self.technique = technique

        if technique == 'dominant':
            self.color_palette_size = int(color_palette_size)

        # TO DO
        # if technique == 'width average':
        #     self.array_height = output_height
            
        self.output_height = output_height
        self.output_width = output_width
        self.nbr_frame_to_compute = output_width
