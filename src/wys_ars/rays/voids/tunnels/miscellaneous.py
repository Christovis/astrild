import sys
import numpy as np

sourceDirectoryPath = "/net/plato/data/users/cautun/bin/python/python_import/C++_code"
includePaths = ["/net/plato/data/users/cautun/Programs/stow/include"]
libraryPaths = ["/net/plato/data/users/cautun/Programs/stow/lib"]

dtype2ctype = {
    np.dtype(np.float64): "double",
    np.dtype(np.float32): "float",
    np.dtype(np.int32): "int",
    np.dtype(np.uint64): "size_t",
    np.dtype(np.int16): "short int",
}


def throwError(
    s1="", s2="", s3="", s4="", s5="", s6="", s7="", s8="", s9="", s10="", EXIT=True
):
    """ Throws an error message and stop the program.
    The function can take up to 10 arguments."""
    print("\n\n~~~ ERROR ~~~ ", s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)
    if EXIT:
        sys.exit(1)


def throwWarning(s1="", s2="", s3="", s4="", s5="", s6="", s7="", s8="", s9="", s10=""):
    """ Throws a warning message but doies not stop the program.
    The function can take up to 10 arguments."""
    print("\n~~~ WARNING ~~~ ", s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)


def ThrowError(s1="", EXIT=True):
    """ Throws an error message and stop the program."""
    print("\n\n~~~ ERROR ~~~ ", s1)
    if EXIT:
        sys.exit(1)


def ThrowWarning(s1=""):
    """ Throws a warning message but doies not stop the program."""
    print("\n~~~ WARNING ~~~ ", s1)


def charToString(charArray):
    """ Takes an array of chars and returns the corresponding string."""
    result = ""
    for i in charArray:
        result += i.decode("UTF-8")
    return result


def BoxLength(box):
    """Returns the box length given the box coordinates."""
    return box[1::2] - box[0::2]


def ValidGridCoordinates(index, grid):
    """Check if an index is a valid grid entry."""
    if ((index >= 0) and (index < grid)).all():
        return True
    else:
        return False


def massInCell(densityHeader):
    """Computes what is the mass of a cell with average density - value=1. The result is returned in units of (M0/h)."""
    rhoCritical = 27.7538e10  # the critical density in units of (M0/h) / (Mpc/h)^3
    length = BoxLength(densityHeader.box)
    volume = length[0] * length[1] * length[2] / densityHeader.totalGrid
    return rhoCritical * densityHeader.Omega0 * volume


def checkArraySizes(shapeArray, nameArray, funcName):
    """Checks that the numpy arrays inserted in the 'array' list have the same dimensions."""
    if len(shapeArray) != len(nameArray):
        print(
            "Could not check that the input arrays to function '%s' have the same dimensions!"
            % funcName
        )
    for i in range(1, len(shapeArray)):
        if shapeArray[0] != shapeArray[i]:
            throwError(
                "The '%s' and '%s' arguments of function '%s' must be numpy arrays having the same shape. The '%s' shape = %s while the '%s' shape = %s."
                % (
                    nameArray[0],
                    nameArray[i],
                    funcName,
                    nameArray[0],
                    str(shapeArray[0]),
                    nameArray[i],
                    str(shapeArray[0]),
                )
            )


def readArrayEntry(array, file, startPosition, noEntries):
    """Reads from file 'fileName' 'noEntries' entries into the array 'array'. The entries are read starting at position 'startPosition' as in a flattened array."""
    maxChunckOfData = 256 ** 3
    noChuncks = int(noEntries / maxChunckOfData)
    for i in range(noChuncks):
        array[startPosition : startPosition + maxChunckOfData] = np.fromfile(
            file, array.dtype, maxChunckOfData
        )
        startPosition += maxChunckOfData
    noLeftOver = int(noEntries - maxChunckOfData * noChuncks)
    array[startPosition : startPosition + noLeftOver] = np.fromfile(
        file, array.dtype, noLeftOver
    )
    return array


def readArrayEntries(array, file, startPosition, noEntries):
    """Reads from file 'fileName' 'noEntries' entries into the array 'array'. The entries are read starting at position 'startPosition' as in a flattened array."""
    maxChunckOfData = 256 ** 3
    noChuncks = int(noEntries / maxChunckOfData)
    for i in range(noChuncks):
        array[startPosition : startPosition + maxChunckOfData] = np.fromfile(
            file, array.dtype, maxChunckOfData
        )
        startPosition += maxChunckOfData
    noLeftOver = int(noEntries - maxChunckOfData * noChuncks)
    array[startPosition : startPosition + noLeftOver] = np.fromfile(
        file, array.dtype, noLeftOver
    )
    return array


def writeArrayEntries(array, file, startPosition, noEntries):
    """Writes to file 'file' 'noEntries' entries from the array 'array'. The entries are written starting at position 'startPosition' as in a flattened array."""
    maxChunckOfData = 256 ** 3
    noChuncks = int(noEntries / maxChunckOfData)
    for i in range(noChuncks):
        temp = array[startPosition : startPosition + maxChunckOfData]
        temp.tofile(file)
        startPosition += maxChunckOfData
    noLeftOver = int(noEntries - maxChunckOfData * noChuncks)
    temp = array[startPosition : startPosition + noLeftOver]
    temp.tofile(file)


def return_raDecDist_coordinates(pos, angleUnit="degree"):
    """Returns spherical coordinates for the given array of positions. You can choose the units of the returned angles to be either 'degree' or 'radian'. """
    dis = np.sqrt((pos * pos).sum(axis=1))
    theta = np.pi / 2.0 - np.arccos(pos[:, 2] / dis)  # the declination
    sinTheta = np.sin(np.arccos(pos[:, 2] / dis))
    phi = np.pi + np.arctan2(
        pos[:, 1] / (dis * sinTheta), pos[:, 0] / (dis * sinTheta)
    )  # the right ascenssion

    if angleUnit in ["degree"]:
        theta *= 180.0 / np.pi
        phi *= 180.0 / np.pi
    elif angleUnit not in ["radian"]:
        throwError(
            "Unknown angle unit in the function 'return_raDecDist_coordinates'. "
        )

    return phi, theta, dis


class Progress:
    """Use to output the progress of a computation."""

    def __init__(self, finalSize):
        self.size = finalSize
        self.count = 0
        self.lastOutput = -1

    def ShowProgress(self, newOutput):
        """Prints the progress on the screen."""
        if self.lastOutput < 0:
            print("\t", newOutput, "%")
        elif newOutput >= 100:
            print("\r\t", newOutput, "%")
        else:
            print("\r\t", newOutput, "%")
        self.lastOutput = newOutput
        sys.stdout.flush()

    def Next(self, step=1):
        """Increases the progress counter by 'step' (DEFAULT=1)."""
        self.count += step
        temp = (self.count * 100) / self.size
        if temp != self.lastOutput:
            self.ShowProgress(temp)

    def Reset(self, finalSize=None):
        """Resets the counter."""
        if finalSize:
            self.size = finalSize
        self.count = 0
        self.lastOutput = -1
