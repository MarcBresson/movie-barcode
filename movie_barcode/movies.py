from typing import List
from tkinter import filedialog, Tk
import logging
import os

from movie_barcode.movie import Movie

logger = logging.getLogger(__name__)

class Movies:
    def __init__(
        self,
    ):
        self.movies: List[Movie] = []



    def load(
        self,
        paths: str | List[str] | None = None,
        dir_input_npy: str = "./npy",
        check_for_existing = True
    ) -> None:
        """
        Will load a movie either from a list of paths, or through a file dialog.

        Parameters
        ~~~~~~~~
        :param int paths:
            Single path or list of path of where to find movies.
        """

        if paths==None:
            root = Tk()
            root.withdraw()
            paths =  filedialog.askopenfilename(
                title = "Select files",
                filetypes = (
                    ("movie files",".mkv .mp4 .mpeg"),
                    ("all files","*.*")
                ),
                multiple=True
            )

        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            self.movies.append(Movie(path))

        # We check if we already computed frames for every movie
        if check_for_existing:
            for movie in self.movies:
                movie.load(dir_input_npy, True)



    def save(
        self,
        dir_output_npy: str = "./npy"
    ) -> None:
        """
        Save the computation of every movies in a file. Defaults to ./npy

        Parameters
        ~~~~~~~~
        :param optional str dir_output_npy:
            Path of where to export the computed frames for each movies.
        """
        os.makedirs(dir_output_npy, exist_ok=True)

        for movie in self.movies:
            movie.save(
                dir_output_npy,
            )



    def compute(
        self,
        id_video_flux: int = 0
    ) -> None:
        """
        Compute frames' dominant colors for each movies.

        Parameters
        ~~~~~~~~
        :param optional str id_video_flux:
            Path of where to export the computed frames for each movies.
        """

        for movie in self.movies:
            movie.compute_colors(id_video_flux)
        
        if len(self.movies)==0:
            logger.error("No movie loaded. Colors generation aborted.")



    def export_barcode(
        self,
        method,
        dir_output_images: str = './images'
    ) -> None:
        """
        Generate the barcode with the desired method.

        Parameters
        ~~~~~~~~
        :param str method:
            Which sorting method to use.
        :param str dir_output_images:
            Where to export results.
        """

        for movie in self.movies:
            movie.export_barcode(
                method,
                dir_output_images
            )



    def export_every_barcode(
        self,
        dir_output_images: str = './images'
    ) -> None:
        """
        Generate the barcode with every methods.

        Parameters
        ~~~~~~~~
        :param str dir_output_images:
            Where to export results.
        """

        for movie in self.movies:
            movie.export_every_barcode(
                dir_output_images
            )
