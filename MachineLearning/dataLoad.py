from osgeo import gdal, gdal_array
import numpy as np

#######################################################################################################################


def train_load(normalize=False):

    # Tell GDAL to throw Python exceptions, and register all drivers
    gdal.UseExceptions()
    gdal.AllRegister()

    # Read in our image
    img_ds = gdal.Open('./MATLAB/2013_IEEE_GRSS_DF_Contest_CASI.tif', gdal.GA_ReadOnly)
    # Printing # of Bands, Columns, and Rows, respectively
    print(' Rows: {row} \n'.format(row=img_ds.RasterYSize),
          'Columns: {cols} \n'.format(cols=img_ds.RasterXSize),
          'Bands: {bands} \n'.format(bands=img_ds.RasterCount))

    # create an empty array with (rows, cols, bands) size, with the same format
    # of the input image.
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    # filling the numpy array
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    normal_img = img

    if normalize:
        normal_img = np.zeros_like(img, np.float32)
        for i in range(img.shape[2]):
            band = img[:, :, i]
            minimum = np.amin(band)
            maximum = np.amax(band)
            normal_img[:, :, i] = np.divide((band - minimum), (maximum - minimum))

    return normal_img

#######################################################################################################################


def roi_load():

    roi_ds = gdal.Open('./ROIsBW.tif', gdal.GA_ReadOnly)
    roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

    return roi

#######################################################################################################################


def envi_load(name: str, subset=False):

    address = './MATLAB/ENVI/' + name
    envi_ds = gdal.Open(address, gdal.GA_ReadOnly)
    envi = np.zeros((envi_ds.RasterYSize, envi_ds.RasterXSize, envi_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(envi_ds.GetRasterBand(1).DataType))

    print(str(envi.shape[2]) + '   bands loaded from   ' + name)
    for b in range(envi.shape[2]):
        envi[:, :, b] = envi_ds.GetRasterBand(b + 1).ReadAsArray()

    if subset:
        s1 = int(input('which n bands do you need?   '))
        s2 = int(input('which n bands do you need?   '))
        envi = envi[:, :, s1:s2]

    normal_envi = np.zeros_like(envi, np.float32)
    for i in range(envi.shape[2]):
        band = envi[:, :, i]
        minimum = np.nanmin(band)
        maximum = np.nanmax(band)
        normal_envi[:, :, i] = np.divide((band - minimum), (maximum - minimum))

    print(normal_envi.shape)
    return normal_envi
