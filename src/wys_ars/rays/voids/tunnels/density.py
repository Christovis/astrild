import os.path
import numpy as np
from wys_ars.rays.voids.tunnels import miscellaneous as misc
from wys_ars.rays.voids.tunnels.miscellaneous import throwError, throwWarning


bufferType = np.uint64
bufferSizeBytes = 8

DENSITY_FILE = 1
VELOCITY_FILE = 11
VELOCITY_GRADIENT_FILE = 12
VELOCITY_DIVERGENCE_FILE = 13
VELOCITY_SHEAR_FILE = 14
VELOCITY_VORTICITY_FILE = 15
VELOCITY_STD_FILE = 16
SCALAR_FIELD_FILE = 20
SCALAR_FIELD_GRADIENT_FILE = 21
GRAVITATIONAL_POTENTIAL_FILE = 50
WATERSHED_FILE = 101

NO_SCALAR_COMPONENTS = 6


densityMethod = {
    1: "DTFE",
    2: "TSC",
    3: "SPH",
    11: "CIC",
    np.uint64(-1): "Unknown",
    -1: "Unknown",
}
densityFileType = {
    DENSITY_FILE: "Density",
    VELOCITY_FILE: "Velocity",
    VELOCITY_GRADIENT_FILE: "Velocity gradient",
    VELOCITY_DIVERGENCE_FILE: "Velocity divergence",
    VELOCITY_SHEAR_FILE: "Velocity shear",
    VELOCITY_VORTICITY_FILE: "Velocity vorticity",
    VELOCITY_STD_FILE: "Velocity standard deviation",
    SCALAR_FIELD_FILE: "Scalar field",
    SCALAR_FIELD_GRADIENT_FILE: "Scalar field gradient",
    GRAVITATIONAL_POTENTIAL_FILE: "gravitational potential",
    WATERSHED_FILE: "Watershed Void Finder void index",
    -1: "Unknown",
}
densityVariableName = {
    DENSITY_FILE: "density",
    VELOCITY_FILE: "velocity",
    VELOCITY_GRADIENT_FILE: "velocityGradient",
    VELOCITY_DIVERGENCE_FILE: "velocityDivergence",
    VELOCITY_SHEAR_FILE: "velocityShear",
    VELOCITY_VORTICITY_FILE: "velocityVorticity",
    VELOCITY_STD_FILE: "velocityStd",
    SCALAR_FIELD_FILE: "scalarField",
    SCALAR_FIELD_GRADIENT_FILE: "scalarFieldGradient",
    GRAVITATIONAL_POTENTIAL_FILE: "gravitationalPotential",
    WATERSHED_FILE: "voidIndex",
    -1: "unknown",
    10001: "oneComponent",
    10002: "twoComponent",
    10003: "threeComponent",
}
densityDataType = {
    DENSITY_FILE: "f4",
    VELOCITY_FILE: "f4",
    VELOCITY_GRADIENT_FILE: "f4",
    VELOCITY_DIVERGENCE_FILE: "f4",
    VELOCITY_SHEAR_FILE: "f4",
    VELOCITY_VORTICITY_FILE: "f4",
    VELOCITY_STD_FILE: "f4",
    SCALAR_FIELD_FILE: "f4",
    SCALAR_FIELD_GRADIENT_FILE: "f4",
    GRAVITATIONAL_POTENTIAL_FILE: "f4",
    WATERSHED_FILE: "i4",
    -1: "f4",
    10001: "f4",
    10002: "f4",
    10003: "f4",
}
densityDataComponents = {
    DENSITY_FILE: 1,
    VELOCITY_FILE: 3,
    VELOCITY_GRADIENT_FILE: 9,
    VELOCITY_DIVERGENCE_FILE: 1,
    VELOCITY_SHEAR_FILE: 5,
    VELOCITY_VORTICITY_FILE: 3,
    VELOCITY_STD_FILE: 1,
    SCALAR_FIELD_FILE: NO_SCALAR_COMPONENTS,
    SCALAR_FIELD_GRADIENT_FILE: 3 * NO_SCALAR_COMPONENTS,
    GRAVITATIONAL_POTENTIAL_FILE: 1,
    WATERSHED_FILE: 1,
    -1: 1,
    10001: 1,
    10002: 2,
    10003: 3,
}


class DensityHeader:
    """ A class used for reading and storing the header of a density grid file. It uses the numpy class to define the variables. """

    byteSize = 1024
    fillSize = byteSize - 13 * 8 - 18 * 8 - 2 * 8

    def __init__(self):
        # variables related to the density computation
        self.gridSize = np.zeros(3, dtype=np.uint64)
        self.totalGrid = np.uint64(0)
        self.fileType = np.int32(-1)
        self.noDensityFiles = np.int32(0)
        self.densityFileGrid = np.zeros(3, dtype=np.int32)
        self.indexDensityFile = np.int32(-1)
        self.box = np.zeros(6, dtype=np.float64)

        # variables from the Gadget snapshot
        self.npartTotal = np.zeros(6, dtype=np.uint64)
        self.mass = np.zeros(6, dtype=np.float64)
        self.time = np.float64(0.0)
        self.redshift = np.float64(0.0)
        self.BoxSize = np.float64(0.0)
        self.Omega0 = np.float64(0.0)
        self.OmegaLambda = np.float64(0.0)
        self.HubbleParam = np.float64(0.0)

        # additional information about files
        self.method = np.uint64(-1)
        self.fill = np.zeros(DensityHeader.fillSize, dtype="c")
        self.FILE_ID = np.int64(1)

    def SetType(self):
        self.gridSize = self.gridSize.astype(np.uint64)
        self.totalGrid = np.uint64(self.totalGrid)
        self.fileType = np.int32(self.fileType)
        self.noDensityFiles = np.int32(self.noDensityFiles)
        self.indexDensityFile = np.int32(self.indexDensityFile)
        self.box = self.box.astype(np.float64)
        self.time = np.float64(self.time)
        self.redshift = np.float64(self.redshift)
        self.BoxSize = np.float64(self.BoxSize)
        self.Omega0 = np.float64(self.Omega0)
        self.OmegaLambda = np.float64(self.OmegaLambda)
        self.HubbleParam = np.float64(self.HubbleParam)
        self.method = np.uint64(self.method)
        self.FILE_ID = np.int64(self.FILE_ID)

    def nbytes(self):
        __size = (
            self.gridSize.nbytes
            + self.totalGrid.nbytes
            + self.fileType.nbytes
            + self.noDensityFiles.nbytes
            + self.densityFileGrid.nbytes
            + self.indexDensityFile.nbytes
            + self.box.nbytes
            + self.npartTotal.nbytes
            + self.mass.nbytes
            + self.time.nbytes
            + self.redshift.nbytes
            + self.BoxSize.nbytes
            + self.Omega0.nbytes
            + self.OmegaLambda.nbytes
            + self.HubbleParam.nbytes
            + self.method.nbytes
            + self.fill.nbytes
            + self.FILE_ID.nbytes
        )
        return __size

    def dtype(self):
        __dt = np.dtype(
            [
                ("gridSize", np.uint64, 3),
                ("totalGrid", np.uint64),
                ("fileType", np.int32),
                ("noDensityFiles", np.uint32),
                ("densityFileGrid", np.uint32, 3),
                ("indexDensityFile", np.uint32),
                ("box", np.float64, 6),
                ("npartTotal", np.uint64, 6),
                ("mass", np.float64, 6),
                ("time", np.float64),
                ("redshift", np.float64),
                ("BoxSize", np.float64),
                ("Omega0", np.float64),
                ("OmegaLambda", np.float64),
                ("HubbleParam", np.float64),
                ("method", np.uint64),
                ("fill", "c", DensityHeader.fillSize),
                ("FILE_ID", np.int64),
            ]
        )
        return __dt

    def TupleAsString(self):
        return "( self.gridSize, self.totalGrid, self.fileType, self.noDensityFiles, self.densityFileGrid, self.indexDensityFile, self.box, self.npartTotal, self.mass, self.time, self.redshift, self.BoxSize, self.Omega0, self.OmegaLambda, self.HubbleParam, self.method, self.fill, self.FILE_ID )"

    def Tuple(self):
        return eval(self.TupleAsString())

    def fromfile(self, f, BUFFER=True, numBytes_data=None):
        if BUFFER:
            __buffer1 = np.fromfile(f, bufferType, 1)[0]
        A = np.fromfile(f, self.dtype(), 1)[0]
        if BUFFER:
            __buffer2 = np.fromfile(f, bufferType, 1)[0]
            if __buffer1 != __buffer2 or __buffer1 != DensityHeader.byteSize:
                throwError(
                    "Error reading the header of the density file. 'buffer1'=",
                    __buffer1,
                    "and 'buffer2'=",
                    __buffer2,
                    "when both should be",
                    DensityHeader.byteSize,
                )
        exec("%s = A" % self.TupleAsString())
        if numBytes_data is not None and BUFFER:
            numBytes_data[:] = np.fromfile(f, bufferType, 1)[0]

    def tofile(self, f, BUFFER=True):
        self.SetType()
        if self.nbytes() != DensityHeader.byteSize:
            throwError(
                "When calling the function 'DensityHeader.tofile()'. The size of the density header is %i while the expected size is %i. Please check which variables do not have the correct size."
                % (self.nbytes(), DensityHeader.byteSize)
            )
        __A = np.array([self.Tuple()], dtype=self.dtype())
        __buffer = np.array([__A.nbytes], dtype=np.uint64)
        if BUFFER:
            __buffer.tofile(f)
        __A.tofile(f)
        if BUFFER:
            __buffer.tofile(f)

    def FileType(self):
        __fileType = "Unknown"
        if self.fileType in densityFileType:
            __fileType = densityFileType[self.fileType]
        return (self.fileType, __fileType)

    def DataType(self):
        """ Returns the type of the data in the file, NOT the type of the header. """
        __dataType = "f4"
        if self.fileType in densityDataType:
            __dataType = densityDataType[self.fileType]
        return __dataType

    def DataComponents(self):
        """ Returns the number of components of the data in the file. """
        __noComponents = 1
        if self.fileType in densityDataComponents:
            __noComponents = densityDataComponents[self.fileType]
        return __noComponents

    def DataName(self):
        """ Returns the name of the variable stored in the file. """
        __dataName = "unknown"
        if self.fileType in densityVariableName:
            __dataName = densityVariableName[self.fileType]
        return __dataName

    def Method(self):
        __method = "Unknown"
        if self.method in densityMethod:
            __method = densityMethod[self.method]
        return (self.method, __method)

    def BoxLength(self):
        __boxLength = np.zeros(self.box.size / 2, self.box.dtype)
        __boxLength[:] = self.box[1::2] - self.box[0::2]
        return __boxLength

    def TotalVolume(self):
        return self.BoxLength().prod()

    def CellVolume(self):
        return self.TotalVolume() / self.totalGrid

    def PrintValues(self):
        print("The values contained in the density header:")
        print("1) Information about the file itself:")
        print("  gridSize        = ", self.gridSize)
        print("  totalGrid       = ", self.totalGrid)
        print("  fileType        = ", self.FileType())
        print("  noDensityFiles  = ", self.noDensityFiles)
        if self.noDensityFiles > 1:
            print("  densityFileGrid = ", self.densityFileGrid)
            print("  indexDensityFile= ", self.indexDensityFile)
        print("  box coordinates = ", self.box)

        print(
            "\n2) Information about the Gadget snapshot used to compute the density:"
        )
        print("  npartTotal   = ", self.npartTotal)
        print("  mass         = ", self.mass)
        print("  time         = ", self.time)
        print("  redshift     = ", self.redshift)
        print("  BoxSize      = ", self.BoxSize)
        print("  Omega0       = ", self.Omega0)
        print("  OmegaLambda  = ", self.OmegaLambda)
        print("  HubbleParam  = ", self.HubbleParam)

        print("\n3) Additional information:")
        print("  method       = ", self.Method())
        print("  FILE_ID      = ", self.FILE_ID)
        print("  fill         = %s" % misc.charToString(self.fill))
        print()

    def AddProgramCommands(self, commands):
        """Adds the program options used to obtain the current results to the 'fill' array in the header."""
        newCommands = self.fill.tostring().rstrip("\x00") + commands + " ;  "
        choice = int(len(newCommands) < DensityHeader.fillSize)
        newLen = [DensityHeader.fillSize, len(newCommands)][choice]
        newOff = [len(newCommands) - DensityHeader.fillSize, 0][choice]
        self.fill[:newLen] = newCommands[newOff:]


def densityMultipleFiles(rootName, fileIndex):
    """ Returns the name of the density file 'fileIndex' when a result is saved in multiple binary files.
    It takes 2 arguments: root name of the files and the file number whose name is requested (from 0 to DensityHeader.noDensityFiles-1)."""
    return rootName + ".%i" % fileIndex


def readDensityHeader(file, VERBOSE=True):
    """   Reads only the density header from the given binary density file. It returns the results as the class 'DensityHeader'.
    Takes as argument the name (or root anme) of the file from where to read the header.
    Can use VERBOSE = False to turn off the message."""
    header = DensityHeader()
    tempName = file
    if not os.path.isfile(tempName):
        tempName = densityMultipleFiles(file, 0)
        if not os.path.isfile(tempName):
            throwError(
                "Cannot find the density binary file. There are no '%s' or '%s' files."
                % (file, tempName)
            )
    if VERBOSE:
        print(
            "Reading the header of the density file '%s' ... " % tempName,
            end=" ",
        )
    f = open(tempName, "rb")
    header.fromfile(f)
    if VERBOSE:
        print("Done")
    f.close()
    return header


def readDensityData(
    file,
    HEADER=True,
    VERBOSE=True,
    wrongFileLabel=False,
    dataType=None,
    dataComponents=None,
):
    """ Reads the data in a density file. It returns a list with the density header (if HEADER=True) and a numpy array with the values of the data at each grid point.
    Use HEADER=False to cancel returning the header and VERBOSE=False to turn off the messages. """

    # read the header and find how many files there are
    header = DensityHeader()
    tempName = file
    if not os.path.isfile(tempName):
        tempName = densityMultipleFiles(file, 0)
        if not os.path.isfile(tempName):
            throwError(
                "Cannot find the density binary file. There are no '%s' or '%s' files."
                % (file, tempName)
            )
    f = open(tempName, "rb")
    numDataBytes = np.zeros(1, np.int64)
    header.fromfile(f, numBytes_data=numDataBytes)
    f.close()
    if header.noDensityFiles > 1:
        for i in range(header.noDensityFiles):
            tempName = densityMultipleFiles(file, i)
            if not os.path.isfile(tempName):
                throwError(
                    "Cannot find the density binary file number %i (of %i files) with expected name '%s'."
                    % (i + 1, header.noDensityFiles, tempName)
                )

    # read the data from each file
    if dataType is None:
        dataType = header.DataType()
    if dataComponents is None:
        dataComponents = header.DataComponents()
    data = np.empty(int(header.totalGrid * dataComponents), dataType)
    if data.nbytes != numDataBytes and header.noDensityFiles == 1:
        throwWarning(
            "There is a problem with the input file. The length in bytes of the data does not match the value expected from the file header!"
        )
        if wrongFileLabel:
            dataComponents = np.int(
                numDataBytes
                * 1.0
                / header.totalGrid
                / np.array([1], dataType).nbytes
            )
            print(dataComponents)
            if (
                numDataBytes
                != dataComponents
                * header.totalGrid
                * np.array([1], dataType).nbytes
            ):
                throwError(
                    "Cannot estimate what is the data format and size of the data array. Please supply these values as input to the read in function via the parameters 'dataType'."
                )
        else:
            throwError(
                "Program stopped since does not know how to read the input file!"
            )
    startPosition = 0

    for i in range(header.noDensityFiles):
        if VERBOSE:
            print(
                "Reading the data in the density file '%s' which is file %i of %i files ... "
                % (tempName, i + 1, header.noDensityFiles)
            )
        tempName = file
        if header.noDensityFiles != 1:
            tempName = densityMultipleFiles(file, i)
        if not os.path.isfile(tempName):
            throwError(
                "Cannot find the density file number %i with expected name '%s'."
                % (i + 1, tempName)
            )
        f = open(tempName, "rb")
        tempHeader = DensityHeader()
        tempHeader.fromfile(f)
        dataSize = np.uint64(tempHeader.totalGrid * dataComponents)

        # reading the data
        __buffer1 = np.fromfile(f, bufferType, 1)[0]
        data = misc.readArrayEntries(data, f, startPosition, dataSize)
        __buffer2 = np.fromfile(f, bufferType, 1)[0]
        if __buffer1 != __buffer2:
            throwError(
                "While reading the density data block in file '%s'." % tempName,
                "The buffers before and after the data do not have the same value (buffer1 = %i, buffer2 = %i while the expected value was %i)."
                % (__buffer1, __buffer2, dataSize * data[0].nbytes),
            )
        startPosition + dataSize

    # return the results
    if HEADER:
        return header, data
    else:
        return data


def writeDensityData(file, header, data, VERBOSE=True):
    """ Writes a density grid data to a binary file which has a Density header and each block of data is preceded and followed by a uint64 integer giving the number of bytes in the data block (for error checking).
    The function takes 3 arguments: name of the output file, the density header (class 'DensityHeader') and the data in the form of a numpy array. """

    # do some error checking
    if VERBOSE:
        print("Writing the density data to the file '%s' ... " % file, end=" ")
    __temp = header.gridSize[0] * header.gridSize[1] * header.gridSize[2]
    if __temp != header.totalGrid:
        throwError(
            "The total number of grid points in the density header is not equal to the product of the grid dimensions along each axis. Total number of grid points is %i while the size along each axis is:"
            % header.totalGrid,
            header.gridSize,
        )
    dataComponents = header.DataComponents()
    noDataElements = data.size / dataComponents
    if header.totalGrid != noDataElements:
        throwError(
            "The total number of grid points in the density header does not agree with the data length. Number grid points in the header is %i while the data has only %i elements."
            % (header.totalGrid, noDataElements)
        )
    header.noDensityFiles = np.int32(1)

    # write the header to file
    f = open(file, "wb")
    header.tofile(f)

    # write the data to file
    data.shape = -1
    noBytes = data.size * data[0].nbytes
    __buffer = np.array([noBytes], dtype=np.uint64)
    __buffer.tofile(f)
    misc.writeArrayEntries(data, f, 0, data.size)
    __buffer.tofile(f)
    f.close()
    if VERBOSE:
        print("Done.")
