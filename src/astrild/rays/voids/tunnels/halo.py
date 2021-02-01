import os.path
import numpy as np
import scipy.spatial

import astrild.rays.voids.tunnels.gadget
import astrild.rays.voids.tunnels.analysis
from astrild.rays.voids.tunnels.miscellaneous import (
    throwError,
    throwWarning,
    dtype2ctype,
    charToString,
    readArrayEntries,
    writeArrayEntries,
)


# labels for integer columns
NUMBER_PARTICLES_LABEL = "npart"
HOST_HALO_LABEL = "hostHalo_2"
NUM_SUBSTRUCTURE_LABEL = "numSubStruct_2"
ENVIRONMENT_LABEL = "environment"
# labels for floating data columns
VIRIAL_RADIUS_LABEL = "Rvir"
ENV_FILA_THICKNESS_LABEL = "fila_thickness"
ENV_FILA_DENSITY_LABEL = "fila_density"
ENV_WALL_THICKNESS_LABEL = "wall_thickness"
ENV_WALL_DENSITY_LABEL = "wall_density"
SHAPE_B_LABEL = "b_shape"
SHAPE_C_LABEL = "c_shape"
# vector labels (contain 3 entries that give the vector projection along the x, y and z directions)
BULK_VELOCITY_LABEL = ["Vcx", "Vcy", "Vcz"]
L_MOMENTUM_LABEL = ["Lx", "Ly", "Lz"]
ENV_FILA_DIRECTION_LABEL = ["fila_dir_x", "fila_dir_y", "fila_dir_z"]
ENV_WALL_DIRECTION_LABEL = ["wall_dir_x", "wall_dir_y", "wall_dir_z"]
SHAPE_A_DIRECTION_LABEL = ["Eax", "Eay", "Eaz"]
SHAPE_B_DIRECTION_LABEL = ["Ebx", "Eby", "Ebz"]
SHAPE_C_DIRECTION_LABEL = ["Ecx", "Ecy", "Ecz"]


class HaloHeader:
    """
    Class that stores and reads the header of the '.halos' binary file. The file has the following format:
    8-byte integer = 1024
    8-byte integer = 1024

    8-byte integer = 16*noColumns       (noColumns = noColumnsIntegers + noColumnsFloats)
    string array with column names (16 characters for each column) -> number of entries = noColumns
    8-byte integer = 16*noColumns

    8-byte integer = data dimensions for integer columns
    integer data columns
    8-byte integer = data dimensions for integer columns

    8-byte integer = data dimensions for floating number columns
    integer data columns
    8-byte integer = data dimensions for floating number columns

    header has the following structure:
    noHalos                 int64       # total number of halos in the file
    noColumnsIntegers       int64       # number of integer properties associated
                                          to each halo
    noColumnsFloats         int64       # number of floating point properties
                                          associated to each halo
    noColumns               int64       # = noColumnsIntegers + noColumnsFloats
    mpcUnit                 float64     # the value of 1 Mpc/h in units of halo
                                          positions (=1 if positions in Mpc, =1000
                                          if positions in kpc)
    box                     6 * float64 # the coordinates of the box:
                                          xMin, xMax, yMin, yMax, zMin, zMax
    positionColumns         3 * int64   # gives the indices of the position
                                          columns in the float data array
    massUnit                float64     # mass unit in M_solar/h
    massRange               2 * float64 # shows minMass and maxMass in the file
    massColumn              int64       # gives the indices of the mass column
    noFiles                 int64       # number of files that store the data
    fill          char until 1024 bytes # keeps additional information in the
                                          float data array
    FILE_ID                 int32       # =100 -> unique ID for halo files
                                          (to distinguish between density
                                           and NEXUS files)
    """

    bufferType = np.uint64
    bufferSizeBytes = 8
    columnNameLength = 16  # the number of chars associated to each column name

    byteSize = 1024
    fillSize = byteSize - 4 * 8 - 10 * 8 - 4 * 8 - 2 * 8

    def __init__(self):
        # variables related to the density computation
        self.noHalos = np.int64(0)
        self.noColumnsIntegers = np.int64(-1)
        self.noColumnsFloats = np.int64(-1)
        self.noColumns = np.int64(-1)

        self.mpcUnit = np.float64(-1.0)
        self.box = np.zeros(6, dtype=np.float64)
        self.positionColumns = np.zeros(3, dtype=np.int64)

        self.massUnit = np.float64(-1.0)
        self.massRange = np.zeros(2, dtype=np.float64)
        self.massColumn = np.int64(-1)

        self.noFiles = np.int64(1)
        self.fill = np.zeros(HaloHeader.fillSize, dtype="c")
        self.FILE_ID = np.int64(100)

    def SetType(self):
        self.noHalos = np.int64(self.noHalos)
        self.noColumnsIntegers = np.int64(self.noColumnsIntegers)
        self.noColumnsFloats = np.int64(self.noColumnsFloats)
        self.noColumns = np.int64(self.noColumns)
        self.mpcUnit = np.float64(self.mpcUnit)
        self.box = np.array(self.box)
        self.massUnit = np.float64(self.massUnit)
        self.massColumn = np.int64(self.massColumn)
        self.noFiles = np.int64(self.noFiles)
        self.FILE_ID = np.int64(self.FILE_ID)

    def nbytes(self):
        __size = (
            self.noHalos.nbytes
            + self.noColumnsIntegers.nbytes
            + self.noColumnsFloats.nbytes
            + self.noColumns.nbytes
            + self.mpcUnit.nbytes
            + self.box.nbytes
            + self.positionColumns.nbytes
            + self.massUnit.nbytes
            + self.massRange.nbytes
            + self.massColumn.nbytes
            + self.noFiles.nbytes
            + self.fill.nbytes
            + self.FILE_ID.nbytes
        )
        return __size

    def dtype(self):
        __dt = np.dtype(
            [
                ("noHalos", np.int64),
                ("noColumnsIntegers", np.int64),
                ("noColumnsFloats", np.int64),
                ("noColumns", np.int64),
                ("mpcUnit", np.float64),
                ("box", np.float64, 6),
                ("positionColumns", np.int64, 3),
                ("massUnit", np.float64),
                ("massRange", np.float64, 2),
                ("massColumn", np.int64),
                ("noFiles", np.int64),
                ("fill", "c", HaloHeader.fillSize),
                ("FILE_ID", np.int64),
            ]
        )
        return __dt

    def TupleAsString(self):
        return "( self.noHalos, self.noColumnsIntegers, self.noColumnsFloats, self.noColumns, self.mpcUnit, self.box, self.positionColumns, self.massUnit, self.massRange, self.massColumn, self.noFiles, self.fill, self.FILE_ID )"

    def Tuple(self):
        return eval(self.TupleAsString())

    def fromfile(self, f, BUFFER=True):
        if BUFFER:
            __buffer1 = np.fromfile(f, HaloHeader.bufferType, 1)[0]
        A = np.fromfile(f, self.dtype(), 1)[0]
        if BUFFER:
            __buffer2 = np.fromfile(f, HaloHeader.bufferType, 1)[0]
            if __buffer1 != __buffer2 or __buffer1 != HaloHeader.byteSize:
                throwError(
                    "Error reading the header of the halo file. 'buffer1'=",
                    __buffer1,
                    "and 'buffer2'=",
                    __buffer2,
                    "when both should be",
                    HaloHeader.byteSize,
                    i,
                )
        exec("%s = A" % self.TupleAsString())

    def tofile(self, f, BUFFER=True):
        self.SetType()
        if self.nbytes() != HaloHeader.byteSize:
            throwError(
                "When calling the function 'HaloHeader.tofile()'. The size of the halo file header is %i while the expected size is %i. Please check which variables do not have the correct size."
                % (self.nbytes(), HaloHeader.byteSize)
            )
        __buffer = np.array([self.nbytes()], dtype=HaloHeader.bufferType)
        __A = np.array([self.Tuple()], dtype=self.dtype())
        if BUFFER:
            __buffer.tofile(f)
        __A.tofile(f)
        if BUFFER:
            __buffer.tofile(f)

    def BoxLength(self):
        __boxLength = np.zeros(self.box.size / 2, self.box.dtype)
        __boxLength[:] = self.box[1::2] - self.box[0::2]
        return __boxLength

    def PrintColumnNames(self):
        print("\tInteger data columns:")
        for i in range(self.columnNamesIntegers.shape[0]):
            print(
                "   %s"
                % ("%i= %s" % (i, charToString(self.columnNamesIntegers[i]))),
                # end=" ",
            )
        print("\n\tFloating data columns:")
        for i in range(self.columnNamesFloats.shape[0]):
            print(
                "   %s"
                % ("%i= %s" % (i, charToString(self.columnNamesFloats[i]))),
                # end=" ",
            )
        print()

    def PrintValues(self, showColumnNames=True):
        print("The values contained in the halo header:")
        print("  noHalos           = ", self.noHalos)
        print("  noColumnsIntegers = ", self.noColumnsIntegers)
        print("  noColumnsFloats   = ", self.noColumnsFloats)
        print("  noColumns         = ", self.noColumns)
        print("  mpcUnit           = ", self.mpcUnit)
        print("  box coordinates   = ", self.box)
        print("  positionColumns   = ", self.positionColumns)
        print("  massUnit          = ", self.massUnit)
        print("  massRange         = ", self.massRange)
        print("  massColumn        = ", self.massColumn)
        print("  noFiles           = ", self.noFiles)
        print("  FILE_ID           = ", self.FILE_ID)
        print("  fill              =  %s" % charToString(self.fill))
        print()
        if showColumnNames:
            self.PrintColumnNames()

    def ReadColumnNames(self, f, BUFFER=True):
        """Reads the column names (16 chars for each column) from a binary halo file. """
        if BUFFER:
            __buffer1 = np.fromfile(f, HaloHeader.bufferType, 1)[0]
        A = np.fromfile(
            f, "c", self.noColumns * HaloHeader.columnNameLength
        ).reshape(-1, HaloHeader.columnNameLength)
        if BUFFER:
            __buffer2 = np.fromfile(f, HaloHeader.bufferType, 1)[0]
            if (
                __buffer1 != __buffer2
                or __buffer1 != self.noColumns * HaloHeader.columnNameLength
            ):
                throwError(
                    "Error reading the column name data stack of the halo file. 'buffer1'=",
                    __buffer1,
                    "and 'buffer2'=",
                    __buffer2,
                    "when both should be",
                    self.noColumns * HaloHeader.columnNameLength,
                )
        self.columnNamesIntegers = A[: self.noColumnsIntegers, :]
        self.columnNamesFloats = A[self.noColumnsIntegers :, :]

    def WriteColumnNames(self, f, BUFFER=True):
        """Writes the column names (16 chars for each column) in a binary halo file. """
        __A = np.hstack(
            (
                self.columnNamesIntegers.flatten(),
                self.columnNamesFloats.flatten(),
            )
        )
        __buffer = np.array([__A.nbytes], dtype=HaloHeader.bufferType)
        if BUFFER:
            __buffer.tofile(f)
        __A.tofile(f)
        if BUFFER:
            __buffer.tofile(f)

    def StringColumnNames(self):
        result = ""
        for i in range(self.columnNamesIntegers.shape[0]):
            result = "%s   %s" % (
                result,
                "%i= %s" % (i, charToString(self.columnNamesIntegers[i])),
            )
        for i in range(self.columnNamesFloats.shape[0]):
            result = "%s   %s" % (
                result,
                "%i= %s" % (i, charToString(self.columnNamesFloats[i])),
            )
        return result

    def AddColumnIntegers(self, columnName):
        """Adds a column name to the 'columnNamesIntegers' array."""
        __A = np.zeros(HaloHeader.columnNameLength, dtype="c")
        __length = [HaloHeader.columnNameLength, len(columnName)][
            int(len(columnName) < HaloHeader.columnNameLength)
        ]
        __A.put(list(range(__length)), columnName)
        self.columnNamesIntegers = np.vstack(
            (self.columnNamesIntegers, __A)
        ).copy()
        self.noColumnsIntegers += 1
        self.noColumns += 1

    def AddColumnFloats(self, columnName):
        """Adds a column name to the 'columnNamesFloats' array."""
        __A = np.zeros(HaloHeader.columnNameLength, dtype="c")
        __length = [HaloHeader.columnNameLength, len(columnName)][
            int(len(columnName) < HaloHeader.columnNameLength)
        ]
        __A.put(list(range(__length)), columnName)
        self.columnNamesFloats = np.vstack((self.columnNamesFloats, __A)).copy()
        self.noColumnsFloats += 1
        self.noColumns += 1

    def NewColumnsIntegers(self, columnNames):
        """Adds new column names for the data stored in the integer columns. The argument "columnNames" needs to be a list of strings of the same size as the number of integer columns."""
        __noIntCols = len(columnNames)
        if __noIntCols != self.noColumnsIntegers:
            throwError(
                "Error in function 'halo.HaloHeader.NewColumnsIntegers'. The length of the list of strings = %i is different than the number of integer columns = %i. The two need to be the same!"
                % (__noIntCols, self.noColumnsIntegers)
            )
        self.columnNamesIntegers = np.zeros(
            (__noIntCols, HaloHeader.columnNameLength), dtype="c"
        )
        for i in range(__noIntCols):
            __length = [HaloHeader.columnNameLength, len(columnNames[i])][
                int(len(columnNames[i]) < HaloHeader.columnNameLength)
            ]
            self.columnNamesIntegers[i, :__length] = columnNames[i][:__length]

    def NewColumnsFloats(self, columnNames):
        """Adds new column names for the data stored in the float columns. The argument "columnNames" needs to be a list of strings of the same size as the number of float columns."""
        __noFloatCols = len(columnNames)
        if __noFloatCols != self.noColumnsFloats:
            throwError(
                "Error in function 'halo.HaloHeader.NewColumnsFloats'. The length of the list of strings = %i is different than the number of float columns = %i. The two need to be the same!"
                % (__noFloatCols, self.noColumnsFloats)
            )
        self.columnNamesFloats = np.zeros(
            (__noFloatCols, HaloHeader.columnNameLength), dtype="c"
        )
        for i in range(__noFloatCols):
            __length = [HaloHeader.columnNameLength, len(columnNames[i])][
                int(len(columnNames[i]) < HaloHeader.columnNameLength)
            ]
            self.columnNamesFloats[i, :__length] = columnNames[i][:__length]

    def GetColumnIndex(self, columnName):
        """
        Returns the index of the column with this given name if it exits.
        It returns: 0/1  columnIndex  True/False
        Where 0=dataIntegers, 1=dataFloats, columnIndex = -1 for not found and
        otherwise the column index. The last return is True if column found and False otherwise.
        """
        intCols, floatCols = [], []
        for i in range(self.columnNamesIntegers.shape[0]):
            intCols.append(
                self.columnNamesIntegers[i].tostring().rstrip("\x00")
            )
        for i in range(self.columnNamesFloats.shape[0]):
            floatCols.append(
                self.columnNamesFloats[i].tostring().rstrip("\x00")
            )

        search = columnName[: HaloHeader.columnNameLength]
        if search in intCols:
            return 0, intCols.index(search), True
        elif search in floatCols:
            return 1, floatCols.index(search), True
        else:
            return -1, -1, False

    def AddProgramCommands(self, commands):
        """
        Adds the program options used to obtain the current results to the 'fill'
        array in the header.
        """
        # newCommands = str(self.fill.tostring()).rstrip("\x00") + str(commands) + " ;  "
        newCommands = str(commands) + " ;  "
        newCommands = list(newCommands)
        choice = int(len(newCommands) < HaloHeader.fillSize)
        newLen = [HaloHeader.fillSize, len(newCommands)][choice]
        newOff = [len(newCommands) - HaloHeader.fillSize, 0][choice]
        self.fill[:newLen] = newCommands[newOff:]


def readHaloHeader(fileName, VERBOSE=True):
    """Reads the header of a halo binary file."""

    # read the header
    if VERBOSE:
        print("Reading the header of the halo file '%s' ... " % (fileName))
    header = HaloHeader()
    f = open(fileName, "rb")
    header.fromfile(f)
    header.ReadColumnNames(f)  # read the column names
    return header


def readHaloData(fileName, VERBOSE=True):
    """Reads the data from a halo binary file. NOTE: reads only 1 halo file, not multiple ones."""

    # read the header
    if VERBOSE:
        print("Reading the data in the halo file '%s' ... " % (fileName))
    header = HaloHeader()
    f = open(fileName, "rb")
    header.fromfile(f)
    header.ReadColumnNames(f)  # read the column names

    # read the integer data
    dataSize = np.uint64(header.noHalos * header.noColumnsIntegers)
    dataIntegers = np.empty(dataSize, np.int32)

    __buffer1 = np.fromfile(f, HaloHeader.bufferType, 1)[0]
    dataIntegers = readArrayEntries(dataIntegers, f, 0, dataSize)
    __buffer2 = np.fromfile(f, HaloHeader.bufferType, 1)[0]
    if __buffer1 != __buffer2:
        throwError(
            "While reading the integer data block in halo file '%s'. The buffer preceding and the buffer after the data do not have the same value (buffer1 = %s, buffer2 = %s while the expected value was %s)."
            % (
                fileName,
                __buffer1,
                __buffer2,
                dataSize * dataIntegers[0].nbytes,
            )
        )

    # read the floating point data
    dataSize = np.uint64(header.noHalos * header.noColumnsFloats)
    dataFloats = np.empty(dataSize, np.float32)

    __buffer1 = np.fromfile(f, HaloHeader.bufferType, 1)[0]
    dataFloats = readArrayEntries(dataFloats, f, 0, dataSize)
    __buffer2 = np.fromfile(f, HaloHeader.bufferType, 1)[0]
    if __buffer1 != __buffer2:
        throwError(
            "While reading the floating point data block in halo file '%s'. The buffer preceding and the buffer after the data do not have the same value (buffer1 = %s, buffer2 = %s while the expected value was %s)."
            % (
                fileName,
                __buffer1,
                __buffer2,
                dataSize * dataIntegers[0].nbytes,
            )
        )

    # return the results
    dataIntegers.shape = header.noHalos, header.noColumnsIntegers
    dataFloats.shape = header.noHalos, header.noColumnsFloats
    return header, dataIntegers, dataFloats


def writeHaloData(fileName, header, dataIntegers, dataFloats, VERBOSE=True):
    """Writes the halo data into a halo binary file."""

    # set some header values before writting the data
    massColumn = header.massColumn
    header.massRange[:] = (
        np.min(dataFloats[:, massColumn]),
        np.max(dataFloats[:, massColumn]),
    )
    if header.noColumnsIntegers * header.noHalos != dataIntegers.size:
        throwError(
            "The integer halo data does not match with the information stored in the halo header. There are %i halo data, but it is expected to be %i data for %i halos with %i integer columns."
            % (
                dataIntegers.size,
                header.noColumnsIntegers * header.noHalos,
                header.noHalos,
                header.noColumnsIntegers,
            )
        )
    if header.noColumnsFloats * header.noHalos != dataFloats.size:
        throwError(
            "The floating point halo data does not match with the information stored in the halo header. There are %i halo data, but it is expected to be %i data for %i halos with %i floating point columns."
            % (
                dataFloats.size,
                header.noColumnsFloats * header.noHalos,
                header.noHalos,
                header.noColumnsFloats,
            )
        )
    header.noColumns = header.noColumnsIntegers + header.noColumnsFloats

    # write the halo header
    if VERBOSE:
        print("Writing the halo data to the binary file '%s' ... " % fileName)
    f = open(fileName, "wb")
    header.tofile(f)
    header.WriteColumnNames(f)  # write the column names

    # write the integer data
    dataIntegers.shape = -1
    noBytes = dataIntegers.size * dataIntegers[0].nbytes
    __buffer = np.array([noBytes], dtype=np.uint64)
    __buffer.tofile(f)
    writeArrayEntries(dataIntegers, f, 0, dataIntegers.size)
    __buffer.tofile(f)

    # write the floating point data
    dataFloats.shape = -1
    noBytes = dataFloats.size * dataFloats[0].nbytes
    __buffer = np.array([noBytes], dtype=np.uint64)
    __buffer.tofile(f)
    writeArrayEntries(dataFloats, f, 0, dataFloats.size)
    __buffer.tofile(f)
    f.close()


def getColumnNames(names, reshuffle=None):
    if isinstance(names, str):
        names = names.split()
    s1 = names
    result = np.zeros((len(s1), HaloHeader.columnNameLength), dtype="c")
    for i in range(len(s1)):
        temp = s1[i].split("(")[0]
        choose = int(len(temp) > HaloHeader.columnNameLength)
        l = [len(temp), HaloHeader.columnNameLength][choose]
        result[i, :l] = temp[:l]
    if not reshuffle:
        return result
    else:
        res = np.zeros(
            (len(list(reshuffle.values())), HaloHeader.columnNameLength),
            dtype="c",
        )
        res[list(reshuffle.values()), :] = result[list(reshuffle.keys()), :]
        return res


def sortHalos(
    dataIntegers,
    dataFloats,
    sortColumn,
    order="descending",
    relabelHalos=False,
    parentHaloColumn=None,
    VERBOSE=True,
):
    """Sorts the halo according to the values in the sorting column 'sortColumn'.
    NOTE: Choose 'relabelHalos'=True to assign halo IDS using the new order in which the halos were sorted.
    """
    funcName = "sortHalos"
    orderValues = ["ascending", "descending"]
    if order not in orderValues:
        throwError(
            "The argument 'order' of function '%s' can take only the values %s."
            % (funcName, str(orderValues))
        )
    if sortColumn.ndim is not 1:
        throwError(
            "The argument 'sortColumn' of function '%s' muts be a 1D numpy array whose values are used for sorting the halo properties."
            % funcName
        )

    if VERBOSE:
        print("Sorting the halos in %s order ..." % (order,))
    newOrder = None
    if order == "ascending":
        newOrder = sortColumn.argsort()
    elif order == "descending":
        newOrder = (-1 * sortColumn).argsort()
    dataIntegers[:] = dataIntegers[newOrder, :]
    dataFloats[:] = dataFloats[newOrder, :]

    if relabelHalos:
        if VERBOSE:
            print("Relabeling the halos according to their new position ...")
        dataIntegers[:, 0] = np.arange(dataIntegers.shape[0])
        if parentHaloColumn:
            if VERBOSE:
                print(
                    "Found substructure ID in halo data: column number %i."
                    % parentHaloColumn
                )
            temp = np.arange(dataIntegers.shape[0])[newOrder.argsort()]
            temp = np.hstack((temp, np.array([-1], np.int64)))
            tempS = dataIntegers[:, parentHaloColumn].flatten()
            dataIntegers[:, parentHaloColumn] = temp[tempS]
        elif VERBOSE:
            print("No information about substructure ID in halo data.")

    return dataIntegers, dataFloats


def gridIndices(haloPos, box, gridSize, VERBOSE=True):
    """Computes the grid indices associated to each halo position. """
    if VERBOSE:
        print("Computing the grid coordinates of the halos ... ")

    offset = box[0::2]  # offset of box coordinates
    dx = (box[1::2] - box[0::2]) / gridSize[:]  # dx for each grid axis
    gridPos = []
    for i in range(offset.shape[0]):
        gridPos.append(((haloPos[:, i] - offset[i]) / dx[i]).astype(np.int32))
    return gridPos


def haloEnvironment(haloPos, envData, box, VERBOSE=True, valueOutsideGrid=-1):
    """Returns an array with the environment tag of each halo. An entry of -1 means that the halo is outside the box with the given coordinates."""
    funcName = "haloEnvironment"
    if len(box) != 6:
        throwError(
            "In function '%s' the argument 'box' must be a 6 elements array. This element gives the coordinates of the environment grid data 'envData'."
            % funcName
        )
    if envData.ndim != 3:
        throwError(
            "In function '%s' the argument 'envData' must be a 3D numpy array."
            % funcName
        )

    if VERBOSE:
        print("Computing the environment of each halo ...")
    dx = (box[1::2] - box[::2]) / envData.shape
    grid = (haloPos[:, :] - box[::2]) / dx[:]
    valid = (
        (grid[:, 0] >= 0)
        * (grid[:, 0] < envData.shape[0])
        * (grid[:, 1] >= 0)
        * (grid[:, 1] < envData.shape[1])
        * (grid[:, 2] >= 0)
        * (grid[:, 2] < envData.shape[2])
    )  # select only halos within the grid
    grid = grid[valid, :]
    grid = grid.astype(np.int32)
    result = np.zeros(haloPos.shape[0], envData.dtype)
    result[~valid] = valueOutsideGrid
    result[valid] = envData[grid[:, 0], grid[:, 1], grid[:, 2]]

    return result


def halosInMassRange(massColumn, minMass, maxMass, VERBOSE=True):
    """ Returns the selection array which has 'True' values for halos with masses in the range 'minMass' to 'maxMass'. The masses are given by the argument 'massColumn'. """
    if VERBOSE:
        print(
            "Selecting the halos with masses in the interval [%e,%e] ... "
            % (minMass, maxMass)
        )
    return (massColumn >= minMass) * (massColumn <= maxMass)


def selectHalosInBox(
    header,
    dataIntegers,
    dataFloats,
    posColumn,
    box,
    periodicLength=None,
    periodic=True,
    VERBOSE=True,
):
    """Selects only the haloes in the box of interest. It uses periodic boundary conditions (if periodic=True) to translate the box to outside its regions if the region of interest is partially outside the periodic box."""
    if VERBOSE:
        print("\nFinding the haloes in the subBox %s ..." % str(box))
    mainBox = header.box
    noHalosInitial = dataIntegers.shape[0]
    if periodicLength is None:
        periodicLength = header.box[1] - header.box[0]
    header.box = box
    pos = dataFloats[:, posColumn]

    # code for when the box is fully contained in the periodic one
    if gadget.boxFullyContained(mainBox, box) or not periodic:
        select = (
            (pos[:, 0] >= box[0])
            * (pos[:, 0] <= box[1])
            * (pos[:, 1] >= box[2])
            * (pos[:, 1] <= box[3])
            * (pos[:, 2] >= box[4])
            * (pos[:, 2] <= box[5])
        )
        header.noHalos = select.sum()
        dataIntegers = dataIntegers[select, :]
        dataFloats = dataFloats[select, :]
        if VERBOSE:
            print(
                "\tfound %i haloes (%.2f%%) out of the total of %i"
                % (
                    header.noHalos,
                    100.0 * header.noHalos / noHalosInitial,
                    noHalosInitial,
                )
            )
        return header, dataIntegers, dataFloats

    # now dealing with the case when the box crosses outside the periodic box
    if VERBOSE:
        print(
            "The subBox in question extends outside the periodic box. Taking this into account."
        )
    n = np.zeros(27, np.int64)
    intersect = np.zeros(27, np.bool)
    select = []
    for i1 in range(-1, 2):
        for i2 in range(-1, 2):
            for i3 in range(-1, 2):
                tempBox = (
                    box[0] + i1 * periodicLength,
                    box[1] + i1 * periodicLength,
                    box[2] + i2 * periodicLength,
                    box[3] + i2 * periodicLength,
                    box[4] + i3 * periodicLength,
                    box[5] + i3 * periodicLength,
                )
                index = (i1 + 1) * 9 + (i2 + 1) * 3 + (i3 + 1)
                intersect[index] = gadget.boxOverlap(mainBox, tempBox)
                if intersect[index]:
                    tempSelect = (
                        (pos[:, 0] >= tempBox[0])
                        * (pos[:, 0] <= tempBox[1])
                        * (pos[:, 1] >= tempBox[2])
                        * (pos[:, 1] <= tempBox[3])
                        * (pos[:, 2] >= tempBox[4])
                        * (pos[:, 2] <= tempBox[5])
                    )
                    n[index] = np.sum(tempSelect)
                    select.append(tempSelect)

    # reserve memory for the output
    if VERBOSE:
        print(
            "\tfound the intersection of the region of interest with %i periodic translations"
            % len(select)
        )
    nTotal = np.sum(n)
    header.noHalos = nTotal
    dInt = np.empty((nTotal, dataIntegers.shape[1]), dataIntegers.dtype)
    dFloat = np.empty((nTotal, dataFloats.shape[1]), dataFloats.dtype)

    # loop again and copy the required particles
    count = 0
    No = 0
    for i1 in range(-1, 2):
        for i2 in range(-1, 2):
            for i3 in range(-1, 2):
                index = (i1 + 1) * 9 + (i2 + 1) * 3 + (i3 + 1)
                if not intersect[index]:
                    continue

                iArray = np.array([i1, i2, i3], dataFloats.dtype)
                dInt[No : No + n[index], :] = dataIntegers[select[count], :]
                dFloat[No : No + n[index], :] = dataFloats[select[count], :]
                dFloat[No : No + n[index], posColumn] -= iArray * periodicLength

                count += 1
                No += n[index]
    if VERBOSE:
        print(
            "\tfound %i haloes (%.2f%%) out of the total of %i"
            % (nTotal, 100.0 * nTotal / noHalosInitial, noHalosInitial)
        )
    return header, dInt, dFloat


def computeStatistics(data, massBins, massBinValues, noBootstrapSamples=100):
    """Computes the statistics for a given distribution of halo properties. The data is a two dimensional numpy array where the first column is the mass of the halo, while the second is the halo property for which the program computes the median, error associated to the median, the 16th and 84th percentile, the average and error of the average."""
    noData = data.shape[1] - 1

    averages = np.zeros((massBins.size - 1, 6 * noData + 2), np.float64)
    averages[:, 0] = massBinValues
    for i in range(
        1, massBins.size
    ):  # loop over the mass bins and get median formation redshif for each bin
        selection = (data[:, 0] >= massBins[i - 1]) * (
            data[:, 0] < massBins[i]
        )  # select only the halos with mass in the given mass bin
        averages[i - 1, 1] = np.sum(selection)
        if averages[i - 1, 1] <= 0:
            continue
        for j in range(0, noData):
            percentile = analysis.Percentile(
                data[selection, j + 1], percentile=(16.0, 84.0), N=1
            )
            averages[i - 1, 6 * j + 2 : 6 * j + 4] = analysis.Median(
                data[selection, j + 1], noBootstrapSamples
            )
            averages[i - 1, 6 * j + 4 : 6 * j + 6] = (
                percentile[0][0],
                percentile[1][0],
            )
            averages[i - 1, 6 * j + 6 : 6 * j + 8] = analysis.Average(
                data[selection, j + 1], noBootstrapSamples
            )
    return averages


def shape_pointCloud(pos, w=1):
    # get the moment of inertia
    x2 = (w * pos[:, 0] * pos[:, 0]).sum()
    y2 = (w * pos[:, 1] * pos[:, 1]).sum()
    z2 = (w * pos[:, 2] * pos[:, 2]).sum()
    xy = (w * pos[:, 0] * pos[:, 1]).sum()
    xz = (w * pos[:, 0] * pos[:, 2]).sum()
    yz = (w * pos[:, 1] * pos[:, 2]).sum()
    inertia = np.array([[x2, xy, xz], [xy, y2, yz], [xz, yz, z2]], np.float64)

    # compute the eigenvalues of the moment of inertia
    inertia /= pos.shape[0]
    eigenValues, eigenVectors = np.linalg.eig(inertia)

    # sort the eigenvalues in decreasing order
    order = eigenValues.argsort()[::-1]

    return np.sqrt(eigenValues[order]), eigenVectors[:, order].transpose()


# Value for the Planck cosmology of P_Millennium
G_halo = 4.302e-9  # newton constant in units of Mpc/h (Msun/h)^-1  (km/s)^2
rhoCrit200_halo = (
    5.54872e13  # 200 times the critical density in units of Msun/h  (Mpc/h)^-3
)


# returns 200 rho_critical given an M200 and R200
def compute200RhoCritical(M200, R200):
    """Returns the 200 rho_critical value given M200 and R200. The mass needs to be in  'Msun/h'  and the radius in  'Mpc/h' ."""
    return 3.0 / (4.0 * np.pi) * M200 / R200 ** 3


# function for returning the virial radius given a virial mass
def virialRadius(M200, rhoCrit200=rhoCrit200_halo):
    """Returns the virial radius in Mpc/h. The mass needs to be in  'Msun/h'  and the critical density in  'Msun/h  (Mpc/h)^-3' ."""
    return (M200 / rhoCrit200 / (4.0 * np.pi / 3.0)) ** (1.0 / 3.0)


# function for returning the virial velocity given a virial mass
def virialVelocity(M200, R200):
    """Returns the virial velocity in km/s. The mass needs to be in  'Msun/h'  and the radius in  'Mpc/h' ."""
    return (G_halo * M200 / R200) ** 0.5
