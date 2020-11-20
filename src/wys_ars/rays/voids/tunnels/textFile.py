import os.path
import numpy as np
from wys_ars.rays.voids.tunnels.miscellaneous import throwError, throwWarning


def readTextFile(
    file,
    noDescriptionLines,
    noRows,
    noColumns,
    dataType,
    delimiter=" ",
    VERBOSE=True,
):
    """ 
    Used to read data from a text file.
    noDescriptionLines - number of description lines.
    noRows - number of rows to read.
    noColumns - number of columns on each row (must give all the columns in the file).
    It returns [description,data], whit data in a numpy matrix of noRows x
    noColumns dimension.
    """
    if not os.path.isfile(file):
        throwError(
            "Could not find the file '%s' for reading (in function 'readTextFile')."
            % file
        )
    if VERBOSE:
        print("Reading the data from the ASCII file '%s' ... " % file, end=" ")
    f = open(file, "r")
    description = ""
    for i in range(noDescriptionLines):
        description += f.readline()
    dataSize = noRows * noColumns
    data = np.fromfile(f, dataType, dataSize, delimiter)
    data.shape = (noRows, noColumns)
    if VERBOSE:
        print("Done")

    return [description, data]


def writeTextFile(file, description=None, data=None, VERBOSE=True):
    if VERBOSE:
        print("Writing the data to the ASCII file '%s' ... " % file)
    f = open(file, "w")
    if description is not None:
        f.write(description)
    if data is not None:
        for i in range(data.shape[0]):
            data[i, :].tofile(f, "  ", "%12.7g")
            f.write("\n")
    f.close()


def getTextFileProperties(
    filename, descriptionChar="#", delimiter=" ", VERBOSE=True
):
    """ Counts the number of description lines, columns and rows in a text file. """
    noDescLines, noColumns, noRows = 0, 0, 1
    if delimiter == " ":
        delimiter = None
    f = open(filename, "r")

    # get number of description lines
    temp = None
    for line in f:
        temp = line.lstrip()
        if len(temp) == 0 or temp[0] == descriptionChar:
            noDescLines += 1
        else:
            break

    # get the number of columns
    noColumns = len(temp.split(delimiter))

    # get the number of rows
    for line in f:
        temp = line.lstrip()
        if len(temp) != 0:
            noRows += 1

    if VERBOSE:
        print(
            "The text file '{0}' has {1} description lines starting with {2}, {3} columns and {4} data rows.".format(
                filename, noDescLines, descriptionChar, noColumns, noRows
            )
        )
    return noDescLines, noColumns, noRows


def writeTextFile_gnuplot3D(file, description, data, VERBOSE=True):
    """ Used to write a 3D data set to a text file.
        file - the name of the output file
        description - description lines in file
        data - the matrix to be written (a numpy array of 1 or 2 dimensions)
    """
    funcName = "writeTextFile_gnuplot3D"
    if data.ndim is not 3:
        throwError(
            "In function '%s' since the data argument (=%iD array) is not a 3D numpy array."
            % (funcName, data.ndim)
        )

    if VERBOSE:
        print(
            "Writing the data to the ASCII file '%s' using the Gnuplot 3D format ... "
            % file,
            end=" ",
        )

    f = open(file, "w")
    f.write(description)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :].tofile(f, "  ", "%12.7g")
            f.write("\n")
        f.write("\n")
    f.close()

    if VERBOSE:
        print("Done")
