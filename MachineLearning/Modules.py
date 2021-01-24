# written by: S. Ali Ahmadi
# last modified: 7/30/2018 - 11:25 PM
#
#
# These modules are written for the purpose of Pattern Recognition course syllabus.
# Some of the functions can be used for general purposes (i.e. read_image)

#
from osgeo import gdal
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
gdal.UseExceptions()


# ######################################################################################################################
def read_image(pathname, filename, normalize=False, stats=False):
    """
    
    :param pathname: directory to the image.
    :param filename: image name with its extension.
    
        examples:
    >>> pathname = './HSI_images/'
    >>> filename = 'image1.tif'
    
    :param normalize: if True, each band will be normalized between 0 & 1.
    :param stats: if True, statistics of each band will be printed.
            also the function will put out a text file containing statistics.
    :return:    read_image(), inputs the filename and directory of an
                image and then puts out the image and its numpy array.
                other options are also available to extract image info.
    NOTE: be careful about the number of rows and columns.
    """
    try:
        data = gdal.Open(pathname + filename, gdal.GA_ReadOnly)
    except RuntimeError:
        print('Unable to open input file')
        sys.exit(1)
    data_array = data.ReadAsArray()

    # Printing # of Bands, Columns, and Rows, respectively
    print(' Rows: {row} \n'.format(row=data.RasterXSize),
          'Columns: {cols} \n'.format(cols=data.RasterYSize),
          'Bands: {bands} \n'.format(bands=data.RasterCount))

    # Further information is available in the following link and other GDAL search results.
    # https: // www.gdal.org / gdal_tutorial.html
    # *****************************************************************************************************************
    if stats:
        # creating a stats file and writing the statistics in it.
        # following lines are changing the output destination of the system
        # from console to the text file.
        # actually, we are writing directly in the file.
        orig_stdout = sys.stdout
        f = open(pathname + 'statistics.txt', 'w')
        sys.stdout = f

        print(' Rows: {row} \n'.format(row=data.RasterXSize),
              'Columns: {cols} \n'.format(cols=data.RasterYSize),
              'Bands: {bands}'.format(bands=data.RasterCount))
        print("No Data Value = NDV \n")

        for band in range(data.RasterCount):
            band += 1
            print("#", band, end="")
            srcband = data.GetRasterBand(band)
            if srcband is None:
                continue
            stats = srcband.GetStatistics(True, True)
            if stats is None:
                continue
            print(" Scale", srcband.GetScale(), end="")
            print(" NDV", srcband.GetNoDataValue())
            print("      Min = %.3f, Max = %.3f, Mean = %.3f, Std = %.3f \n" %
                  (stats[0], stats[1], stats[2], stats[3]))

        for band in range(data.RasterCount):
            band += 1
            srcband = data.GetRasterBand(band)
            stats = srcband.GetStatistics(True, True)
            print("%d, %.3f, %.3f, %.3f, %.3f" %
                  (band, stats[0], stats[1], stats[2], stats[3]))

        sys.stdout = orig_stdout
        f.close()

    # *****************************************************************************************************************
    if normalize:
        temp = data_array.copy()/1000
        data_array = np.zeros_like(temp, np.float32)
        for i in range(temp.shape[0]):
            # the shape axis is different due to the order in which GDAL reads data.
            band = temp[i, :, :]
            minimum = np.amin(band)
            maximum = np.amax(band)
            data_array[i, :, :] = np.divide((band - minimum), (maximum - minimum))

    return data, data_array.T


# ######################################################################################################################
def array_to_raster(array, pathname, filename, src_file=None):
    """
    
    :param array: The input array that is going to be written on the disk as 
            GDAL raster. The raster has parameters such as projection, geo-
            transform, etc. These parameters are taken from the source file 
            or if not specified, are set to zero.
    :param pathname: Folder to which the file is going to save.
    :param filename: Name of the output raster with its format.
    :param src_file: An optional file. If specified, the geographic information will
            be taken from it; if not, the geo information will be set to default.
    :return: function returns the output raster in GDAL format and writes it
              to the disk.
    """
    dst_filename = pathname + filename

    rows = array.shape[0]
    cols = array.shape[1]
    if array.ndim == 3:
        num_bands = array.shape[2]
    else:
        num_bands = 1

    # *****************************************************************************************************************
    if src_file:

        geo_transform = src_file.GetGeoTransform()
        projection = src_file.GetProjectionRef()  # Projection

        # Need a driver object. By default, we use GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        outfile = driver.Create(dst_filename, xsize=cols, ysize=rows,
                                bands=num_bands, eType=gdal.GDT_Float32)
        outfile.SetGeoTransform(geo_transform)
        outfile.SetProjection(projection)

        if array.ndim == 3:
            for b in range(num_bands):
                outfile.GetRasterBand(b + 1).WriteArray(array[:, :, b].astype(np.float32))
        else:
            outfile.GetRasterBand(1).WriteArray(array.astype(np.float32))

    else:

        # Need a driver object. By default, we use GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        outfile = driver.Create(dst_filename, xsize=cols, ysize=rows,
                                bands=num_bands, eType=gdal.GDT_Float32)

        if array.ndim == 3:
            for b in range(num_bands):
                outfile.GetRasterBand(b + 1).WriteArray(array[:, :, b].astype(np.float32))
        else:
            outfile.GetRasterBand(1).WriteArray(array.astype(np.float32))

    return outfile


# ######################################################################################################################
def read_roi(pathname, filename, separate=False, percent=0.7):
    """
    
    :param pathname: directory to the ROI image.
    :param filename: image name with its extension.
    :param separate: if True, it means that test/train ROI files are not separated
            and they should be created from the original file; so the file will be
            split into two files with a specified split percent.
    :param percent: specifies the split percentage for test and train data.
    :return: outputs the ROI image in uint8 type. Also the labels of the classes
             are exported in the labels variable.
             
             *** (it should be completed to return ROIs ready for machine learning)
             *** (maybe in another function like, sort_roi)
    """
    roi_ds = gdal.Open(pathname+filename, gdal.GA_ReadOnly)
    roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    labels = np.unique(roi[roi > 0])
    print('Train/Test data includes {n} classes: {classes}'
          .format(n=labels.size, classes=labels))
    n_samples = (roi > 0).sum()
    print('It has {n} samples.'.format(n=n_samples))

    # *****************************************************************************************************************
    if separate:
        train = np.zeros_like(roi)
        test = np.zeros_like(roi)
        for l in labels:
            cls = roi == l  # looping through classes
            np.put(train, np.random.permutation(np.flatnonzero(cls))[0:int(len(np.flatnonzero(cls)) * percent)], l)
            np.put(test, np.random.permutation(np.flatnonzero(cls))[int(len(np.flatnonzero(cls)) * percent)+1:], l)

        array_to_raster(train, pathname, 'PySeparateTrain.tif', roi_ds)
        array_to_raster(test, pathname, 'PySeparateTest.tif', roi_ds)

    # printing train /test files information into text files.
    orig_stdout = sys.stdout
    f = open(pathname + 'GroundTruth.txt', 'w')
    sys.stdout = f

    print("---------number of samples information.")
    unique_elements, counts_elements = np.unique(roi, return_counts=True)
    print(unique_elements)
    for l in range(len(labels)):
        print("class %d has %d samples." % (l, counts_elements[l]))

    sys.stdout = orig_stdout
    f.close()

    return roi.T, labels


# ######################################################################################################################
def spectrum_plot(data_array, roi, labels):
    """
    
    :param data_array: the output array from read_image module which is 
            a numpy array. note that this array is transposed in previous 
            steps.
    :param roi: the ROI image from read_roi module.
    :param labels: output of the read_roi module which indicates the DN values 
            of the samples of each class from ROI image.
    :return: the function does not return any specific value; but it shows the 
            a subplot containing spectral curves of all classes.  
    """
    plt.figure()
    plt.rc('font', size=8)
    plt.suptitle('spectral reflectance curve of training samples in each class',
                 fontsize=15, fontname={'serif'})

    for c in range(labels.size):
        x = data_array[roi == labels[c], :]
        plt.subplot(3, 5, c + 1)

        for b in range(0, x.shape[0], 5):
            plt.scatter(range(x.shape[1]), x[b, :], marker='.', color='k',
                        s=0.3, alpha=0.6)
            plt.tick_params(axis='y', length=3.0, pad=1.0, labelsize=7)
            plt.tick_params(axis='x', length=0, labelsize=0)

    plt.show()


# ######################################################################################################################
def difference(time1, time2, channel=0, datype=float):
    """
    
    :param time1: image of time 1, which is pre-phenomena;
    :param time2: image of time 2, which is post-phenomena; sizes must
            agree each other.
    :param channel: the default value is 0, which means all the bands;
            but the user can specify to perform the application only on 
            one specific channel (band) of the image.
    :param datype: The default value for data type is considered to be
           float; but user can change to any acceptable data type they want.
    :return: the function computes the difference of two images for
             two different times. The difference image and its size
             are two outputs of the function.
    """

    # checking for array sizes to be matched.
    try:
        np.shape(time1) == np.shape(time2)
    except ValueError:
        print('Input images are not the same size or /n does not have same number of bands.')

    # changing data type to what the user wants to be.
    if datype is float:
        time1.astype(float)
        time2.astype(float)
    else:
        time1.astype(datype)
        time2.astype(datype)

    numbands = np.shape(time1)[0]
    # computing difference map from both images.
    if channel is 0:
        # default case is switched. function will use all the bands.
        diff_image = np.zeros_like(time1)
        for i in range(numbands):
            diff_image[i, :, :] = time2[i, :, :] - time1[i, :, :]
    else:
        diff_image = time2[channel, :, :] - time1[channel, :, :]

    print(np.shape(diff_image))

    return diff_image


# ######################################################################################################################
def pca_transform(reshaped_array, n):
    """
    
    :param reshaped_array: an array with the shape of (n_samples, n_features).
    :param n: number of principle components that should remain after transformation. 
    :return: a new array obtained from PCA with the shape of (n_samples, n_components)
    """

    pca = PCA(n_components=n)
    new_array = pca.fit_transform(reshaped_array)

    return new_array


# ######################################################################################################################

