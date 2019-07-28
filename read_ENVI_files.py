    def read_image(self, filename: str, f_format='.tif', reshape=False):
        """

            :param pathname: string, {directory to the image}
            :param filename: string, {image name with its extension}
            :param f_format: string, ('.tif' by default)
                    input image format which could be one of 3
                    '.tif'/'.tiff' or '.mat' or '.npy'. the default format
                    is '.tif' which is imported via GDAL.
            :param reshape: bool, (False by default)
                    if reshape is True, input file will be reshaped into
                    [n_samples, n_features] shape.
        
            :return: 
                    read_image(), inputs the filename and directory of an
                    image and then puts out the image numpy array and its
                    reshaped array.
        
            Raises:
            -------
            RuntimeError
                If gdal is not able to read the file under any circumstances,
                a runtime error will be raised and may cause the program to exit.

            NOTE: BE CAREFUL about the number of rows and columns.
            """

        import sys
        from scipy.io import loadmat
        from osgeo import gdal, gdal_array
        gdal.UseExceptions()
        import numpy as np

        print('>>     Reading the data...   ')
        # %%%%%%%%%%%%%%%%%%%%%%%%% INTERIOR CONDITION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # check file format
        if f_format == '.tif' or f_format == '.tiff':
            try:
                data = gdal.Open(self.pathname + filename, gdal.GA_ReadOnly)
            except RuntimeError:
                print('GDAL: Unable to open input file')
                sys.exit(1)
            data_array = data.ReadAsArray()
            print('> Tiff image successfully read by GDAL. <')
            
        # %%%%%%%%%%%%%%%%%%%%%%%%% INTERIOR CONDITION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Reading ENVI files. These files have a .hdr file next to themselves.
        elif f_format == '':
            try:
                data = gdal.Open(self.pathname + filename, gdal.GA_ReadOnly)
            except RuntimeError:
                print('GDAL: Unable to open input file')
                sys.exit(1)
            data_array = np.zeros((data.RasterYSize, data.RasterXSize, data.RasterCount),
                                  dtype=gdal_array.GDALTypeCodeToNumericTypeCode(data.GetRasterBand(1).DataType))
            for b in range(data_array.shape[2]):
                # Remember, GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
                data_array[:, :, b] = data.GetRasterBand(b + 1).ReadAsArray()
            print('> ENVI file successfully loaded by GDAL. <')
        

	# Printing # of Bands, Columns, and Rows, respectively
        row = data_array.T.shape[0]
        cols = data_array.T.shape[1]
        if np.ndim(data_array) > 2:
            bands = data_array.T.shape[2]
            print(' Rows: {row} \n'.format(row=row),
                  'Columns: {cols} \n'.format(cols=cols),
                  'Bands: {bands} \n'.format(bands=bands))
        else:
            bands = 1
            print(' Rows: {row} \n'.format(row=row),
                  'Columns: {cols} \n'.format(cols=cols))

        # %%%%%%%%%%%%%%%%%%%%%%%%% INTERIOR CONDITION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # reshape data into [n_samples, n_features]
        if reshape:
            print('     >>   Reshaping...    ')
            data_reshaped = np.reshape(data_array.T, (row * cols, bands), order='F')
            print('     >    Reshaped array size is {size} <'.format(size=[row * cols, bands]))
            out_dict = {'reshaped_data': data_reshaped, 'data_size': data_array.shape}
        # %%%%%%%%%%%%%%%%%%%%%%%%% INTERIOR CONDITION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else:
            # if reshape='False', read_image will return None.
            out_dict = None

        return out_dict, data_array.T



pathname = 'YOUR DIRECTORY/Flood/'
filename1 = 'GLSTN_FL_ROI_1'
filename2 = 'GLSTN_FL_ROI_2'
pm_data = pm.Data(pathname=pathname)
_, multi1 = pm_data.read_image(filename=filename1, f_format='', reshape=False)
_, multi2 = pm_data.read_image(filename=filename2, f_format='', reshape=False)

plt.figure()
for i in range(8):
    image = (multi1[i, ...] - multi1[i, ...].min()) / \
            (multi1[i, ...].max() - multi1[i, ...].min())
    plt.subplot(2, 4, i+1)
    plt.axis('off')
    plt.title('Band {}'.format(i+1))
    plt.imshow(image.T, cmap='gray', clim=(0.0, 0.15))
plt.show()

ndvi1 = (multi1[4, :, :] - multi1[3, :, :])/(multi1[4, :, :] + multi1[3, :, :])
ndvi2 = (multi2[4, :, :] - multi2[3, :, :])/(multi2[4, :, :] + multi2[3, :, :])
ndvi1 = np.multiply(ndvi1 < 1, ndvi1)
ndvi2 = np.multiply(ndvi2 < 1, ndvi2)
plt.figure(), plt.imshow(ndvi1.T, cmap='Greens'), plt.axis('off')
plt.title('1st time NDVI'), plt.show()
plt.figure(), plt.imshow(ndvi2.T, cmap='Greens'), plt.axis('off')
plt.title('2nd time NDVI'), plt.show()
plt.figure(), plt.imshow(ndvi1.T - ndvi2.T, cmap='jet')
plt.title('Change map: Flood area in red'), plt.show()