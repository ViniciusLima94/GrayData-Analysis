"""
Class to generate animations of matrices (3D) based on matplotlib.
Inspired on the tutorial by Tomek:
https://labs.filestack.com/posts/pyplot-animations/
"""
import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

from tqdm import tqdm


class ANIMESHOW():
    """ Create animation of matrix of size NxNxT,
    where T corresponds to the number of time frames """

    def __init__(self, matrix: xr.DataArray, name: str = None,
                 verbose: bool = False):
        """
            Constructor method

            Parameters:
            ----------
            matrix: xarray.DataArray
                Matrix containing the data.
            name:
                name of the movie.
        """

        # Attributin name
        if name is None:
            self.name = 'frame'
        else:
            self.name = name

        # Get verbose level
        self.verbose = verbose

        # Should be a 3d tensor
        assert matrix.ndim == 3
        assert 'times' in matrix.dims

        # Get matrix as class att
        self.matrix = matrix
        # Number of time frame
        self.n_times = matrix.shape[-1]
        self.times = matrix.times.data

        # Path to store the frames
        self.movie_dir = "tmp/frames"
        if not os.path.exists(self.movie_dir):
            os.makedirs(self.movie_dir)
        else:
            files = glob.glob(f'{self.movie_dir}/*')
            for f in files:
                os.remove(f)

    def save_frames(self, figsize=None, dpi=None, im_args={}):
        # Open the matplotlib figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Define storage location for the frames
        frame_storage_path = self.movie_dir + '/{}.png'

        _itr = range(self.n_times)
        for frame_id in tqdm(_itr) if self.verbose else _itr:
            # Prepare the data and plot
            ax.imshow(self.matrix[..., frame_id], aspect="auto", **im_args)
            plt.title(f'{self.times[frame_id]}')

            # Pad the name with 0s so alphabetical and num. orders are the same
            file_name = f'{self.name}_{frame_id:05}'
            file_path = frame_storage_path.format(file_name)

            # Write the current frame to disk
            fig.savefig(file_path)

            # Remove everything from the figure
            ax.clear()
        plt.close()

    def ffmpeg_movie(self, framerate: int = 30):
        movie_dir = Path(self.movie_dir)
        image_pattern = Path(movie_dir) / f'{self.name}_*'
        savepath = Path(f"tmp/{self.name}").with_suffix('.mp4')

        command = f"""
            ffmpeg -y -framerate {framerate} -f image2 -pattern_type glob \
            -i '{image_pattern}' -c:v libx264 -r 30 -profile:v high -crf 20 \
            -pix_fmt yuv420p {savepath}
        """
        subprocess.call(command, shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

    def movie_to_gif(self, ):
        movie_path = f"tmp/{self.name}.mp4"
        gif_path = f"tmp/{self.name}.gif"

        command = f'ffmpeg -y -i {movie_path} -f gif {gif_path}'
        subprocess.call(command, shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

    def create_movie(self, figsize=None, dpi=None, framerate=30, im_args={}):
        self.save_frames(figsize=figsize, dpi=dpi, im_args=im_args)
        self.ffmpeg_movie(framerate=framerate)
        self.movie_to_gif()
