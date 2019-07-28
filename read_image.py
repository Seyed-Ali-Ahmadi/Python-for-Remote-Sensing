def read_image(pathname: str, filename: str, f_format='.tif'):
    """

        :param pathname: string, {directory to the image}
        :param filename: string, {image name with its extension}
        :param f_format: string, ('.tif' by default)
                input image format. The default format
                is '.tif' which is imported via GDAL.
    
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
    from osgeo import gdal
    gdal.UseExceptions()
    import numpy as np

    # check file format
    if f_format == '.tif' or f_format == '.tiff':
        try:
            data = gdal.Open(self.pathname + filename, gdal.GA_ReadOnly)
        except RuntimeError:
            print('GDAL: Unable to open input file')
            sys.exit(1)
        data_array = data.ReadAsArray()
        print('> Tiff image successfully read by GDAL. <')

    else:
        print('--->      Input file format is not valid.     <----')
        sys.exit(1)

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

    return data_array.T
