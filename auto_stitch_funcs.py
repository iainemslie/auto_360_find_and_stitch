import os
import tifffile
import numpy as np
import multiprocessing as mp
from functools import partial

class AutoStitchFunctions:
    def __init__(self, parameters):
        self.lvl0 = os.path.abspath(parameters["input_dir"])
        self.ct_dirs = []
        self.ct_list = []
        self.z_dirs = {}
        self.parameters = parameters

    def run_auto_stitch(self):
        """
        Main function that calls all other functions
        """
        self.print_parameters()

        # Check input directory and find structure
        print("--> Finding CT Directories")
        self.find_ct_dirs()

        # Get the names of the CT "sample" directories
        self.get_ct_list()
        print(self.ct_list)

        # Create a dict with each CTDir as a key and a list of its subdirectories as the value
        self.get_z_dirs()
        print(self.z_dirs)

        # Find 0 and 180 degree pairs and compute the centre
        self.find_images_and_compute_centre()




    def find_ct_dirs(self):
        """
        Walks directories rooted at "Input Directory" location
        Appends their absolute path to ctdir if they contain a directory with same name as "tomo" entry in GUI
        """
        for root, dirs, files in os.walk(self.lvl0):
            for name in dirs:
                if name == "tomo":
                    self.ct_dirs.append(root)
        self.ct_dirs = list(set(self.ct_dirs))

    def get_ct_list(self):
        """
        Creates a list containing the "sample" or CT directory names
        """
        if len(self.ct_dirs) == 0:
            print("--> No valid CT Directories found - Please select a different input directory")
        else:
            for path in self.ct_dirs:
                ct_path, z_dir = os.path.split(path)
                parent_path, ct_name = os.path.split(ct_path)
                self.ct_list.append(ct_name)
            self.ct_list = sorted(list(set(self.ct_list)))

    def get_z_dirs(self):
        """
        Creates a dictionary where each key is a CTDir and its value is a list of its subdirectories
        """
        try:
            for ct_dir in self.ct_list:
                zdir_list = os.listdir(self.parameters['input_dir'] + "/" + ct_dir)
                for zdir in zdir_list:
                    if os.path.isfile(os.path.join(self.parameters['input_dir'], ct_dir, zdir)):
                        zdir_list.remove(zdir)
                self.z_dirs[ct_dir] = sorted(zdir_list)
        except FileNotFoundError:
            print("File Not Found Error")

    def find_images_and_compute_centre(self):
        ct_items = self.z_dirs.items()
        for ct_dir in ct_items:
            for zdir in ct_dir[1]:
                # Get list of image names in the directory
                try:
                    tmp_path = os.path.join(self.parameters['input_dir'], ct_dir[0], zdir, "tomo")
                    image_list = sorted(os.listdir(tmp_path))
                    num_images = len(image_list)
                    # Get the images corresponding to 0 and 180 degree rotations in half-acquisition mode
                    zero_degree_image_name = image_list[0]
                    one_eighty_degree_image_name = image_list[int(num_images / 2) - 1]

                    zero_degree_image_path = os.path.join(tmp_path, zero_degree_image_name)
                    one_eighty_degree_image_path = os.path.join(tmp_path, one_eighty_degree_image_name)

                    print("--> " + str(zdir))
                    print(zero_degree_image_path)
                    print(one_eighty_degree_image_path)

                    self.compute_center(zero_degree_image_path, one_eighty_degree_image_path)

                except NotADirectoryError:
                    print("Skipped - Not a Directory: " + tmp_path)

    def compute_center(self, zero_degree_image_path, one_eighty_degree_image_path):
        # Read each image into a numpy array
        with tifffile.TiffFile(zero_degree_image_path) as tif:
            first = tif.pages[0].asarray().astype(np.float)
        with tifffile.TiffFile(one_eighty_degree_image_path) as tif:
            second = tif.pages[-1].asarray().astype(np.float)

        # Do flat field correction on the images
        flat_files = self.get_filtered_filenames(self.parameters['flats_dir'])
        dark_files = self.get_filtered_filenames(self.parameters['darks_dir'])
        flats = np.array([tifffile.TiffFile(x).asarray().astype(np.float) for x in flat_files])
        darks = np.array([tifffile.TiffFile(x).asarray().astype(np.float) for x in dark_files])
        dark = np.mean(darks, axis=0)
        flat = np.mean(flats, axis=0) - dark
        first = (first - dark) / flat
        second = (second - dark) / flat

        # We must crop the first image from first pixel column up until overlap
        #first_cropped = first[:, :int(self.parameters['overlap_region'])]
        # We must crop the second image from the overlap until the last pixel column
        #second_cropped = second[:, int(self.parameters['overlap_region']):]

        axis = self.compute_rotation_axis(first, second)

        print(str(int(axis)))

    def get_filtered_filenames(self, path, exts=['.tif', '.edf']):
        result = []

        try:
            for ext in exts:
                result += [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]
        except OSError:
            return []

        return sorted(result)

    def compute_rotation_axis(self, first_projection, last_projection):
        """
        Compute the tomographic rotation axis based on cross-correlation technique.
        *first_projection* is the projection at 0 deg, *last_projection* is the projection
        at 180 deg.
        """
        from scipy.signal import fftconvolve
        width = first_projection.shape[1]
        first_projection = first_projection - first_projection.mean()
        last_projection = last_projection - last_projection.mean()

        # The rotation by 180 deg flips the image horizontally, in order
        # to do cross-correlation by convolution we must also flip it
        # vertically, so the image is transposed and we can apply convolution
        # which will act as cross-correlation
        convolved = fftconvolve(first_projection, last_projection[::-1, :], mode='same')
        center = np.unravel_index(convolved.argmax(), convolved.shape)[1]

        return (width / 2.0 + center) / 2

    def print_parameters(self):
        """
        Prints parameter values with line formatting
        """
        print()
        print("**************************** Running Auto Stitch ****************************")
        print("======================== Parameters ========================")
        print("Input Directory: " + self.parameters['input_dir'])
        print("Output Directory: " + self.parameters['output_dir'])
        print("Temp Directory: " + self.parameters['temp_dir'])
        print("Flats Directory: " + self.parameters['flats_dir'])
        print("Darks Directory: " + self.parameters['darks_dir'])
        print("Overlap Region Size: " + self.parameters['overlap_region'])
        print("Number of Steps: " + self.parameters['steps'])
        print("Axis on left: " + self.parameters['axis_on_left'])
        print("============================================================")

    """****** BORROWED FUNCTIONS ******"""

    def read_image(self, file_name):
        """Read tiff file from disk by :py:mod:`tifffile` module."""
        with tifffile.TiffFile(file_name) as f:
            return f.asarray(out='memmap')

    '''
    def open_images_and_stitch(self, ax, crop, first_image_path, second_image_path, out_fmt):
        # We pass index and formats as argument
        first = self.read_image(first_image_path)
        second = self.read_image(second_image_path)
        # We flip the second image before stitching
        second = np.fliplr(second)
        stitched = self.stitch(first, second, ax, crop)
        tifffile.imsave(out_fmt, stitched)
    '''
    '''
    def stitch(self, first, second, axis, crop):
        h, w = first.shape
        if axis > w / 2:
            dx = int(2 * (w - axis) + 0.5)
        else:
            dx = int(2 * axis + 0.5)
            tmp = np.copy(first)
            first = second
            second = tmp
        result = np.empty((h, 2 * w - dx), dtype=first.dtype)
        ramp = np.linspace(0, 1, dx)
        # Mean values of the overlapping regions must match, which corrects flat-field inconsistency
        # between the two projections
        k = np.mean(first[:, w - dx:]) / np.mean(second[:, :dx])
        second = second * k
        result[:, :w - dx] = first[:, :w - dx]
        result[:, w - dx:w] = first[:, w - dx:] * (1 - ramp) + second[:, :dx] * ramp
        result[:, w:] = second[:, dx:]
        return result[:, slice(int(crop), int(2 * (w - axis) - crop), 1)]
    '''



    '''
    def create_temp_dir(self):
        """
        Creates the temp directory and its subdirectories
        The range directory contains directories named in sequence
        from the start of the horizontal overlap range until the end
        """
        try:
            os.mkdir(self.parameters['temp_dir'])
            print("--> Creating Temp Directory: " + self.parameters['temp_dir'])
            ct_items = self.z_dirs.items()
            for ct_dir in ct_items:
                os.makedirs(os.path.join(self.parameters['temp_dir'], ct_dir[0]))
                for z_dir in ct_dir[1]:
                    os.makedirs(os.path.join(self.parameters['temp_dir'], ct_dir[0], z_dir, "projections"))
                    os.makedirs(os.path.join(self.parameters['temp_dir'], ct_dir[0], z_dir, "range"))
                    for num in range(self.overlap_range + 1):
                        tmp_path = os.path.join(self.parameters['temp_dir'], ct_dir[0],
                                                z_dir, "range", str(int(self.parameters['overlap_start']) + num))
                        os.makedirs(os.path.join(tmp_path, "tomo"))
                        os.makedirs(os.path.join(tmp_path, "darks"))
                        os.makedirs(os.path.join(tmp_path, "flats"))
        except FileExistsError:
            print("--> Directory " + self.parameters['temp_dir'] +
                  " already exists - select a different temp directory or delete the current one")
    def find_and_stitch_images(self):
        """
        Gets the images corresponding to 0 degrees and 180 degrees.
        In a dataset of 6000 images - we want 0 and 2999 stitched for 0 degrees - and 3000 and 5999 for 180 degrees
        Also gets the full list of flats/darks and stitches those in pairs
        """
        ct_items = self.z_dirs.items()
        for ct_dir in ct_items:
            for zdir in ct_dir[1]:
                # Get list of image names in the directory
                try:
                    tmp_path = os.path.join(self.parameters['input_dir'], ct_dir[0], zdir, "tomo")
                    image_list = sorted(os.listdir(tmp_path))
                    num_images = len(image_list)
                    # Get the images corresponding to 0 and 180 degree rotations in half-acquisition mode
                    first_zero_degree_image = image_list[0]
                    second_zero_degree_image = image_list[int(num_images/2)-1]
                    first_180_degree_image = image_list[int((num_images/2))]
                    second_180_degree_image = image_list[num_images-1]
                    # Get absolute paths for these images
                    first_zero_degree_image_path = os.path.join(tmp_path, first_zero_degree_image)
                    second_zero_degree_image_path = os.path.join(tmp_path, second_zero_degree_image)
                    first_180_degree_image_path = os.path.join(tmp_path, first_180_degree_image)
                    second_180_degree_image_path = os.path.join(tmp_path, second_180_degree_image)
                    # Get the list of images in flats, darks
                    tmp_flat_path = os.path.join(self.parameters['input_dir'], ct_dir[0], zdir, "flats")
                    flats_list = sorted(os.listdir(os.path.join(tmp_flat_path)))
                    tmp_dark_path = os.path.join(self.parameters['input_dir'], ct_dir[0], zdir, "darks")
                    darks_list = sorted(os.listdir(os.path.join(tmp_dark_path)))
                    # For each axis value in overlap range we stitch corresponding images and save to temp directory
                    pool = mp.Pool(processes=mp.cpu_count())
                    index = range(self.overlap_range + 1)
                    exec_func = partial(self.stitch_fdt, ct_dir, zdir, first_zero_degree_image_path, second_zero_degree_image_path,
                                        first_180_degree_image_path, second_180_degree_image_path,
                                        flats_list, tmp_flat_path, darks_list, tmp_dark_path)
                    pool.map(exec_func, index)
                except NotADirectoryError:
                    print("Skipped - Not a Directory: " + tmp_path)
    def stitch_fdt(self, ct_dir, zdir, first_zero_degree_image_path, second_zero_degree_image_path,
                   first_180_degree_image_path, second_180_degree_image_path, flats_list, tmp_flat_path,
                   darks_list, tmp_dark_path, index):
        """
        Stitches together the input flats/darks/tomo images and outputs to temp directory
        """
        rotation_axis = int(self.parameters['overlap_start']) + index
        out_path = os.path.join(self.parameters['temp_dir'], ct_dir[0],
                                zdir, "range", str(rotation_axis))
        slice_zero_path = os.path.join(out_path, "tomo", "Sli-0.tif")
        slice_180_path = os.path.join(out_path, "tomo", "Sli-180.tif")
        self.open_images_and_stitch(rotation_axis, 0, first_zero_degree_image_path,
                                    second_zero_degree_image_path, slice_zero_path)
        self.open_images_and_stitch(rotation_axis, 0, first_180_degree_image_path,
                                    second_180_degree_image_path, slice_180_path)
        # Stitch pairs of images together - for 20 images we stitch 0-10, 1-11, ..., 9-19
        flat_midpoint = int(len(flats_list) / 2)
        for flat_index in range(flat_midpoint):
            first_flat_path = os.path.join(tmp_flat_path, flats_list[flat_index])
            second_flat_path = os.path.join(tmp_flat_path, flats_list[flat_index + flat_midpoint])
            flat_out_path = os.path.join(out_path, "flats", "Flat_stitched_{:>04}.tif".format(flat_index))
            self.open_images_and_stitch(rotation_axis, 0, first_flat_path, second_flat_path, flat_out_path)
        dark_midpoint = int(len(darks_list) / 2)
        for dark_index in range(dark_midpoint):
            first_dark_path = os.path.join(tmp_dark_path, darks_list[dark_index])
            second_dark_path = os.path.join(tmp_dark_path, darks_list[dark_index + dark_midpoint])
            dark_out_path = os.path.join(out_path, "darks", "Dark_stitched_{:>04}.tif".format(dark_index))
            self.open_images_and_stitch(rotation_axis, 0, first_dark_path, second_dark_path, dark_out_path)
    def flat_field_correction(self):
        """
        Get flats/darks/tomo paths in temp directory and call tofu flat correction
        """
        ct_items = self.z_dirs.items()
        for ct_dir in ct_items:
            print(ct_dir[0])
            for zdir in ct_dir[1]:
                print("-->" + zdir)
                temp_path = os.path.join(self.parameters['temp_dir'], ct_dir[0], zdir, "range")
                range_list = os.listdir(temp_path)
                for index in range_list:
                    try:
                        index_path = os.path.join(temp_path, index)
                        tomo_path = os.path.join(index_path, "tomo")
                        flats_path = os.path.join(index_path, "flats")
                        darks_path = os.path.join(index_path, "darks")
                        # Flat correct image using darks and flats - save tomo/ffc
                        cmd = 'tofu flatcorrect --fix-nan-and-inf --output-bytes-per-file 0'
                        cmd += ' --projections {}'.format(tomo_path + "/Sli-0.tif")
                        cmd += ' --flats {}'.format(flats_path)
                        cmd += ' --darks {}'.format(darks_path)
                        cmd += ' --output {}'.format(os.path.join(temp_path, 'ffc', str(index) + '-sli-0.tif'))
                        os.system(cmd)
                        cmd = 'tofu flatcorrect --fix-nan-and-inf --output-bytes-per-file 0'
                        cmd += ' --projections {}'.format(tomo_path + "/Sli-180.tif")
                        cmd += ' --flats {}'.format(flats_path)
                        cmd += ' --darks {}'.format(darks_path)
                        cmd += ' --output {}'.format(os.path.join(temp_path, 'ffc', str(index) + '-sli-180.tif'))
                        os.system(cmd)
                    except NotADirectoryError:
                        print("Skipped - Not a Directory: " + index_path)
    def subtract_images(self):
        """
        For each pair of 0 and 180 degree images. Flip the 180 degree image around the vertical axis
        Subtract 180 degree image from the 0 degree image
        Save the images to the projections directory
        """
        overlap_start = int(self.parameters['overlap_start'])
        ct_items = self.z_dirs.items()
        for ct_dir in ct_items:
            for zdir in ct_dir[1]:
                temp_path = os.path.join(self.parameters['temp_dir'], ct_dir[0], zdir, "range", "ffc")
                for num in range(self.overlap_range + 1):
                    # Create paths for corresponding 0 and 180 degree images - and for the output subtracted image
                    image_0_path = os.path.join(temp_path, str(overlap_start + num) + "-sli-0.tif")
                    image_180_path = os.path.join(temp_path, str(overlap_start + num) + "-sli-180.tif")
                    subtracted_image_path = os.path.join(self.parameters['temp_dir'], ct_dir[0],
                                                         zdir, "projections",
                                                         "sli-" + str(overlap_start + num) + ".tif")
                    # Open corresponding 0 and 180 degree images
                    image_0 = tifffile.imread(image_0_path)
                    image_180 = tifffile.imread(image_180_path)
                    # Flip the 180 degree image left-right (around "Cartesian y-axis")
                    flipped_180_image = np.fliplr(image_180)
                    # Subtract flipped 180 degree image from 0 degree image
                    subtracted_image = np.subtract(flipped_180_image, image_0)
                    # Save the subtracted image
                    tifffile.imwrite(subtracted_image_path, subtracted_image)
    '''