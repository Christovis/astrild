import os.path
import numpy as np
from wys_ars.rays.voids.tunnels import analysis

# ~ from scipy.weave import inline, converters
from wys_ars.rays.voids.tunnels.miscellaneous import (
    throwError,
    throwWarning,
    charToString,
    readArrayEntries,
    writeArrayEntries,
    sourceDirectoryPath,
    includePaths,
    libraryPaths,
    checkArraySizes,
)

currentDirectory = sourceDirectoryPath
bufferType = np.uint64
bufferSizeBytes = 8


# the different types of MMF environments
MMF_NODE = 4
MMF_FILAMENT = 3
MMF_WALL = 2
MMF_ALL = 5


# the different types of MMF response output
MMF_RESPONSE = 1  # contains the response value at a given scale
MMF_EIGEN = 5  # contains the eigenvalues and eigenvectors for a given scale
MMF_EIGENVECTOR = 6  # contains the significant eigenvectors for a given features (eigenvector 3 for filaments -the direction along the filament- and eigenvector 1 for wall -the direction perpendicular to the wall-)
MMF_MAX_RESPONSE = 10  # contains the values of the maximum MMF response
MMF_MAX_RESPONSE_SCALE = 11  # contains the values of the maximum MMF response and also the scale corresponding to the maximum response
MMF_MAX_EIGEN = (
    15  # the eigenvalues and eigenvectors corresponding to the maximum response scale
)
MMF_MAX_EIGENVECTOR = 16  # same as 6, but corresponding to the maximum response scale
MMF_CLEAN_RESPONSE = 20  # contains a short int data with values of 0 or 1, depending if the pixel is a valid feature or not (e.g. for filaments: 1=pixel is part of a filament, 0=pixel is not part of the filament)
MMF_CLEAN_RESPONSE_COMBINED = 21  # contains the above information, but for all the environments: 0 = void, 2 = wall, 3 = filament and 4 = node
MMF_OBJECTS = 30  # all the pixels corresponding to the same object have the same value, the object tag/id
MMF_DIRECTIONS = 40  # contains the directions for the valid fialment/wall cells
MMF_SPINE = 41  # contains the spine position for each valid fialment/wall cells
MMF_PROPERTIES = 50  # contains the properties (thickness and mass density) for the valid fialment/wall cells
MMF_INDIVIDUAL_SPLIT = 51  # gives the object ID, mean angle and angle std for the operation of splitting the structures into individual objects


MMFFeature = {4: "node", 3: "filament", 2: "wall", -1: "unknown", 5: "all environments"}
MMFFilterTypes = {
    40: "node filter 40 (from Miguel thesis)",
    41: "node filter 41",
    42: "node filter 42",
    43: "node filter 43",
    30: "filament filter 30 (from Miguel thesis)",
    31: "filament filter 31",
    32: "filament filter 32",
    33: "filament filter 33",
    20: "wall filter 20 (from Miguel thesis)",
    21: "wall filter 21",
    22: "wall filter 22",
    23: "wall filter 23",
    -1: "Unknown",
    50: "multiple filters since combined file",
}
MMFFileType = {
    1: "response",
    5: "response eigenvalues & eigenvectors",
    6: "response eigenvectors giving direction for filaments and walls",
    10: "maximum response",
    11: "maximum response & scale",
    15: "maximum response eigenvalues & eigenvectors",
    16: "maximum response eigenvectors giving direction for filaments and walls",
    20: "MMF clean response",
    21: "MMF combined clean response",
    30: "MMF objects",
    40: "filament/wall directions",
    41: "spine position",
    50: "filament/wall properties (thickness and mass density)",
    51: "filament/wall individual split file (object ID, angle mean and std)",
    -1: "Unknown",
}
MMFVariableName = {
    1: "response",
    5: "eigenvaluesEigenvectors",
    6: "environmentEigenvectors",
    10: "maxResponse",
    11: "maxResponse",
    15: "maxEigenvaluesEigenvectors",
    16: "maxEnvironmentEigenvectors",
    20: "cleanResponse",
    21: "combinedCleanResponse",
    30: "MMFObjects",
    40: "MMFDirections",
    41: "MMFSpines",
    50: "MMFProperties",
    51: "MMFIndividualSplit",
    -1: "unknown",
}
MMFDataType = {
    1: "f4",
    5: "f4",
    6: "f4",
    10: "f4",
    11: "f4",
    15: "f4",
    16: "f4",
    20: "i2",
    21: "i2",
    30: "i4",
    40: "f4",
    41: "f4",
    50: "f4",
    51: "f4",
    -1: "f4",
}
MMFDataComponents = {
    1: 1,
    2: 3,
    5: 9,
    6: 3,
    10: 1,
    11: 3,
    15: 9,
    16: 3,
    20: 1,
    21: 1,
    30: 1,
    40: 1,
    41: 1,
    50: 1,
    51: 1,
    -1: 1,
}
MMFMethod = {
    1: "density",
    100: "logarithm filtering of density",
    5: "density logarithm",
    10: "gravitational potential",
    20: "velocity divergence",
    25: "velocity divergence logarithm",
    30: "velocity potential",
    -1: "Unknown",
}


class MMFHeader:
    """ A class used for reading and storing the header of a binary MMF grid file. It uses the numpy class to define the variables. """

    byteSize = 1024
    fillSize = byteSize - 16 * 8 - 18 * 8 - 8

    def __init__(self):
        # variables related to the density computation
        self.gridSize = np.zeros(3, dtype=np.uint64)
        self.totalGrid = np.uint64(0)
        self.feature = np.int32(-10)
        self.scale = np.int32(-10)
        self.radius = np.float32(-1.0)
        self.bias = np.float32(-1.0)
        self.filter = np.int32(-1)
        self.fileType = np.int32(-1)
        self.noMMFFiles = np.int32(1)
        self.MMFFileGrid = np.zeros(3, dtype=np.int32)
        self.indexMMFFile = np.int32(-1)
        self.method = np.int32(-1)
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
        self.fill = np.zeros(MMFHeader.fillSize, dtype="c")
        self.FILE_ID = np.int64(10)

    def SetType(self):
        self.totalGrid = np.uint64(self.totalGrid)
        self.feature = np.int32(self.feature)
        self.scale = np.int32(self.scale)
        self.radius = np.float32(self.radius)
        self.bias = np.float32(self.bias)
        self.filter = np.int32(self.filter)
        self.fileType = np.int32(self.fileType)
        self.noMMFFiles = np.int32(self.noMMFFiles)
        self.indexMMFFile = np.int32(self.indexMMFFile)
        self.method = np.int32(self.method)
        self.time = np.float64(self.time)
        self.redshift = np.float64(self.redshift)
        self.BoxSize = np.float64(self.BoxSize)
        self.Omega0 = np.float64(self.Omega0)
        self.OmegaLambda = np.float64(self.OmegaLambda)
        self.HubbleParam = np.float64(self.HubbleParam)
        self.FILE_ID = np.int64(self.FILE_ID)

    def nbytes(self):
        __size = (
            self.gridSize.nbytes
            + self.totalGrid.nbytes
            + self.feature.nbytes
            + self.scale.nbytes
            + self.radius.nbytes
            + self.bias.nbytes
            + self.filter.nbytes
            + self.fileType.nbytes
            + self.noMMFFiles.nbytes
            + self.MMFFileGrid.nbytes
            + self.indexMMFFile.nbytes
            + self.method.nbytes
            + self.box.nbytes
            + self.npartTotal.nbytes
            + self.mass.nbytes
            + self.time.nbytes
            + self.redshift.nbytes
            + self.BoxSize.nbytes
            + self.Omega0.nbytes
            + self.OmegaLambda.nbytes
            + self.HubbleParam.nbytes
            + self.fill.nbytes
            + self.FILE_ID.nbytes
        )
        return __size

    def dtype(self):
        __dt = np.dtype(
            [
                ("gridSize", np.uint64, 3),
                ("totalGrid", np.uint64),
                ("feature", np.int32),
                ("scale", np.int32),
                ("radius", np.float32),
                ("bias", np.float32),
                ("filter", np.int32),
                ("fileType", np.int32),
                ("noMMFFiles", np.int32),
                ("MMFFileGrid", np.int32, 3),
                ("indexMMFFile", np.int32),
                ("method", np.int32),
                ("box", np.float64, 6),
                ("npartTotal", np.uint64, 6),
                ("mass", np.float64, 6),
                ("time", np.float64),
                ("redshift", np.float64),
                ("BoxSize", np.float64),
                ("Omega0", np.float64),
                ("OmegaLambda", np.float64),
                ("HubbleParam", np.float64),
                ("fill", "c", MMFHeader.fillSize),
                ("FILE_ID", np.int64),
            ]
        )
        return __dt

    def TupleAsString(self):
        return "( self.gridSize, self.totalGrid, self.feature, self.scale, self.radius, self.bias, self.filter, self.fileType, self.noMMFFiles, self.MMFFileGrid, self.indexMMFFile, self.method, self.box, self.npartTotal, self.mass, self.time, self.redshift, self.BoxSize, self.Omega0, self.OmegaLambda, self.HubbleParam, self.fill, self.FILE_ID )"

    def Tuple(self):
        return eval(self.TupleAsString())

    def fromfile(self, f, BUFFER=True):
        if BUFFER:
            __buffer1 = np.fromfile(f, bufferType, 1)[0]
        A = np.fromfile(f, self.dtype(), 1)[0]
        if BUFFER:
            __buffer2 = np.fromfile(f, bufferType, 1)[0]
            if __buffer1 != __buffer2 or __buffer1 != MMFHeader.byteSize:
                throwError(
                    "Error reading the header of the MMF file. 'buffer1'=",
                    __buffer1,
                    "and 'buffer2'=",
                    __buffer2,
                    "when both should be",
                    MMFHeader.byteSize,
                )
        exec("%s = A" % self.TupleAsString())

    def tofile(self, f, BUFFER=True):
        self.SetType()
        if self.nbytes() != MMFHeader.byteSize:
            throwError(
                "When calling the function 'MMFHeader.tofile()'. The size of the MMF header is %i while the expected size is %i. Please check which variables do not have the correct size."
                % (self.nbytes(), MMFHeader.byteSize)
            )
        __buffer = np.array([self.nbytes()], dtype=np.uint64)
        __A = np.array([self.Tuple()], dtype=self.dtype())
        if BUFFER:
            __buffer.tofile(f)
        __A.tofile(f)
        if BUFFER:
            __buffer.tofile(f)

    def Feature(self):
        __feature = "Unknown"
        if self.feature in MMFFeature:
            __feature = MMFFeature[self.feature]
        return (self.feature, __feature)

    def FilterType(self):
        __filterType = "Unknown"
        if self.filter in MMFFilterTypes:
            __filterType = MMFFilterTypes[self.filter]
        return (self.filter, __filterType)

    def FileType(self):
        """ Returns the type of data stored in the file. """
        __fileType = "Unknown"
        if self.fileType in MMFFileType:
            __fileType = MMFFileType[self.fileType]
        return (self.fileType, __fileType)

    def DataType(self):
        """ Returns the type of the data in the file, NOT the type of the header. """
        __dataType = "f4"
        if self.fileType in MMFDataType:
            __dataType = MMFDataType[self.fileType]
        return __dataType

    def DataComponents(self):
        """ Returns the  number of components of the data in the file. """
        __noComponents = 1
        if self.fileType in MMFDataComponents:
            __noComponents = MMFDataComponents[self.fileType]
        return __noComponents

    def DataName(self):
        """ Returns the name of the variable stored in the file. """
        __dataName = "unknown"
        if self.fileType in MMFVariableName:
            __dataName = MMFVariableName[self.fileType]
        return __dataName

    def Method(self):
        """ Returns the method used to compute the MMF data. """
        __method = "Unknown"
        if self.method in MMFMethod:
            __method = MMFMethod[self.method]
        return (self.method, __method)

    def BoxLength(self):
        __boxLength = np.zeros(self.box.size / 2, self.box.dtype)
        __boxLength[:] = self.box[1::2] - self.box[0::2]
        return __boxLength

    def PrintValues(self):
        print("The values contained in the MMF header:")
        print("1) Information about the file itself:")
        print("  gridSize        = ", self.gridSize)
        print("  totalGrid       = ", self.totalGrid)
        print("  feature         = ", self.Feature())
        print("  scale           = ", self.scale)
        print("  radius          = ", self.radius)
        print("  bias            = ", self.bias)
        print("  filter          = ", self.FilterType())
        print("  fileType        = ", self.FileType())
        print("  noMMFFiles      = ", self.noMMFFiles)
        if self.noMMFFiles > 1:
            print("  MMFFileGrid     = ", self.MMFFileGrid)
            print("  indexMMFFile    = ", self.indexMMFFile)
        print("  method          = ", self.Method())
        print("  box coordinates = ", self.box)

        print("\n2) Information about the Gadget snapshot used to compute the density:")
        print("  npartTotal   = ", self.npartTotal)
        print("  mass         = ", self.mass)
        print("  time         = ", self.time)
        print("  redshift     = ", self.redshift)
        print("  BoxSize      = ", self.BoxSize)
        print("  Omega0       = ", self.Omega0)
        print("  OmegaLambda  = ", self.OmegaLambda)
        print("  HubbleParam  = ", self.HubbleParam)

        print("\n3) Additional information:")
        print("  FILE_ID      = ", self.FILE_ID)
        print("  fill         = %s" % charToString(self.fill))
        print()

    def AddProgramCommands(self, commands):
        """Adds the program options used to obtain the current results to the 'fill' array in the header."""
        newCommands = self.fill.tostring().rstrip("\x00") + commands + " ;  "
        choice = int(len(newCommands) < MMFHeader.fillSize)
        newLen = [MMFHeader.fillSize, len(newCommands)][choice]
        newOff = [len(newCommands) - MMFHeader.fillSize, 0][choice]
        self.fill[:newLen] = newCommands[newOff:]


def MMFMultipleFiles(rootName, fileIndex):
    """ Returns the name of the MMF file 'fileIndex' when a result is saved in multiple binary files.
    It takes 2 arguments: root name of the files and the file number whose name is requested (from 0 to MMFHeader.noDensityFiles-1)."""
    return rootName + ".%i" % fileIndex


def readMMFHeader(file, VERBOSE=True):
    """ Reads only the MMF header from the given binary file. It returns the results as the class 'MMFHeader'.
    Takes as argument the name (or root anme) of the file from where to read the header.
    Can use VERBOSE = False to turn off the message."""
    header = MMFHeader()
    tempName = file
    if not os.path.isfile(tempName):
        tempName = MMFMultipleFiles(file, 0)
        if not os.path.isfile(tempName):
            throwError(
                "Cannot find the MMF binary file. There are no '%s' or '%s' files."
                % (file, tempName)
            )
    if VERBOSE:
        print("Reading the header of the MMF file '%s' ... " % tempName, end=" ")
    f = open(tempName, "rb")
    header.fromfile(f)
    f.close()
    if VERBOSE:
        print("Done")
    return header


def readMMFData(file, HEADER=True, VERBOSE=True):
    """ Reads the data in a MMF file. It returns a list with the MMF header (if HEADER=True) and a numpy array with the values of the data at each grid point.
    Use HEADER=False to cancel returning the header and VERBOSE=False to turn off the messages. """

    # read the header and find how many files there are
    header = MMFHeader()
    tempName = file
    if not os.path.isfile(tempName):
        tempName = MMFMultipleFiles(file, 0)
        if not os.path.isfile(tempName):
            throwError(
                "Cannot find the MMF binary file. There are no '%s' or '%s' files."
                % (file, tempName)
            )
    f = open(tempName, "rb")
    header.fromfile(f)
    f.close()
    if header.noMMFFiles > 1:
        for i in range(header.noMMFFiles):
            tempName = MMFMultipleFiles(file, i)
            if not os.path.isfile(tempName):
                throwError(
                    "Cannot find the MMF binary file number %i (of %i files) with expected name '%s'."
                    % (i + 1, header.noMMFFiles, tempName)
                )

    # read the data from each file
    dataType = header.DataType()
    dataComponents = np.uint64(header.DataComponents())
    data = np.empty(header.totalGrid * dataComponents, dataType)
    startPosition = 0

    for i in range(header.noMMFFiles):
        if VERBOSE:
            print(
                "Reading the data in the MMF file '%s' which is file %i of %i files ... "
                % (tempName, i + 1, header.noMMFFiles)
            )
        tempName = file
        if header.noMMFFiles != 1:
            tempName = MMMFMultipleFiles(file, i)
        if not os.path.isfile(tempName):
            throwError(
                "Cannot find the MMF file number %i with expected name '%s'."
                % (i + 1, tempName)
            )
        f = open(tempName, "rb")
        tempHeader = MMFHeader()
        tempHeader.fromfile(f)
        dataSize = np.uint64(tempHeader.totalGrid * dataComponents)

        # reading the data
        __buffer1 = np.fromfile(f, bufferType, 1)[0]
        data = readArrayEntries(data, f, startPosition, dataSize)
        __buffer2 = np.fromfile(f, bufferType, 1)[0]
        if __buffer1 != __buffer2:
            throwError(
                "While reading the MMF data block in file '%s'. The buffer preceding and the buffer after the data do not have the same value (buffer1 = %s, buffer2 = %s while the expected value was %s)."
                % (tempHeader, __buffer1, __buffer2, dataSize * data[0].nbytes)
            )

    # return the results
    results = []
    if HEADER:
        results += [header]
    results += [data]
    return results


def writeMMFData(file, header, data, VERBOSE=True, ERROR_CHECK=True):
    """ Writes MMF data to a binary file which has a MMF header and each block of data is preceded and followed by a uint64 integer giving the number of bytes in the data block (for error checking).
    The function takes 3 arguments: name of the output file, the MMF header (class 'MMFHeader') and the data in the form of a numpy array. """

    # do some error checking
    if VERBOSE:
        print("Writing the MMF data to the file '%s' ... " % file, end=" ")
    __temp = header.gridSize[0] * header.gridSize[1] * header.gridSize[2]
    if __temp != header.totalGrid and ERROR_CHECK:
        throwError(
            "The total number of grid points in the MMF header is not equal to the product of the grid dimensions along each axis. Total number of grid points is %i while the size along each axis is:"
            % header.totalGrid,
            header.gridSize,
        )
    dataComponents = header.DataComponents()
    noDataElements = data.size / dataComponents
    if header.totalGrid != noDataElements and ERROR_CHECK:
        throwError(
            "The total number of grid points in the MMF header does not agree with the data length. Number grid points in the header is %i while the data has only %i elements."
            % (header.totalGrid, noDataElements)
        )
    header.noMMFFiles = np.int32(1)

    # write the header to file
    f = open(file, "wb")
    header.tofile(f)

    # write the data to file
    data.shape = -1
    noBytes = data.size * data[0].nbytes
    __buffer = np.array([noBytes], dtype=np.uint64)
    __buffer.tofile(f)
    writeArrayEntries(data, f, 0, data.size)
    __buffer.tofile(f)
    f.close()
    if VERBOSE:
        print("Done.")


def getHeaderType(file):
    """Checks the type of known binary files. It can recognize: 'Gadget', 'Density', 'MMF', 'Halo' file types. """
    headerType = "Unknown"

    # check for Gadget snapshot type 1 & 2
    buffer1 = np.fromfile(file, np.int32, 1)
    swappedBuffer = buffer1.byteswap()[0]
    if (
        buffer1[0] == 256
        or buffer1[0] == 8
        or swappedBuffer == 256
        or swappedBuffer == 8
    ):
        headerType = "Gadget"
        return headerType

    # check if this is a Density, MMF or Halo file
    FILE_ID = {1: "Density", 10: "MMF", 100: "Halo"}
    buffer1 = np.fromfile(file, np.uint64, 1)[0]
    if buffer1 != 1024:
        return headerType  # returns 'Unknown'
    header = readMMFHeader(file, VERBOSE=False)
    if header.FILE_ID in list(FILE_ID.keys()):
        return FILE_ID[
            header.FILE_ID
        ]  # some files don't have a valid value for this argument, so proceed further
    elif (
        header.filter / 10 == header.feature
    ):  # header.filter//10==header.feature only for MMF files
        headerType = "MMF"
    elif header.gridSize.prod() == header.totalGrid:
        headerType = "Density"
    elif header.gridSize[1:3].sum() == header.totalGrid:
        headerType = "Halo"
    return headerType


def MMFMask(Mask):
    """ Takens in the clean response and returns a int mask with -1 for invalid cells and 0 for valid cells. """
    mask = Mask.astype(np.int32)
    mask[mask == 0] = -1
    mask[mask != -1] = 0
    return mask


# ~ def identifyMMFObjects(Mask,minSize=0,neighborType=1,cellVolume=1.,VERBOSE=True):
# ~ """ This function returns the MMF objects labeled in descending order according to their size (volume).
# ~ It returns the mask with the objects labeled uniquely according to their volume, with the larger objects first. It also returns the volume of each object.
# ~ """
# ~
# ~ if VERBOSE: print "Computing all the objects larger than %i grid cells ... " % minSize
# ~
# ~ # C++ code that does the neighbor searching and pruning of the results
# ~ identifyMMFObjectsSupportCode = ''' #include "%s/MMF_computations.cc" ''' % currentDirectory
# ~ identifyMMFObjectsCode = '''
# ~ #line 1000 "identifyMMFObjectsCode"
# ~
# ~ //get the unique MMF objects
# ~ blitz::Array<int,1> tempObjectSize(2);
# ~ identifyDistinctObjects( mask, noNeighbors, &tempObjectSize );
# ~
# ~ //relabel the object according to increasing size in volume
# ~ int noObjects = relabelMMFObjects( mask, tempObjectSize, minSize );
# ~ objectSize = tempObjectSize;
# ~
# ~ return_val = noObjects;
# ~ '''
# ~
# ~ # compute the objects now
# ~ mask = MMFMask(Mask)
# ~ noNeighbors = (6,26)[int(neighborType==2)]
# ~ objectSize = np.zeros( mask.size/100, np.int32 )
# ~
# ~ noObjects = inline( identifyMMFObjectsCode, ['mask','minSize', 'noNeighbors','objectSize'], type_converters=converters.blitz, support_code=identifyMMFObjectsSupportCode, libraries=['gsl','gslcblas', 'm'], extra_compile_args=['-O3 '], force=0 ) #-fopenmp -DENABLE_OPENMP ,extra_link_args=['-lgomp']
# ~ objectSize = np.resize( objectSize, noObjects ).astype(np.float32)
# ~ if VERBOSE: print "\t found %i objects with volumes from %i to %i grid cells" % (noObjects,objectSize[0],objectSize[-1])
# ~ spacing = np.zeros(3,np.float32)
# ~ objectSize *= cellVolume
# ~
# ~ return (mask,objectSize)
# ~
# ~
# ~ def MMFObjectsMass(mask,density,boxLength,VERBOSE=True):
# ~ """ Computes the mass in each MMF object:
# ~ mask = an integer numpy array giving the label of the MMF object (-1=no object, 0..n=the label of the objects)
# ~ density = a float numpy array giving the density in each cell of the mask
# ~ boxLength = the box length along each dimension
# ~ """
# ~
# ~ if VERBOSE: print "Computing the mass of each MMF object ..."
# ~ # Use C++ to get the mass
# ~ MMFObjectsMassCode = '''
# ~ #line 1000 "MMFObjectsMassCode"
# ~ int const NX = mask.extent(0), NY = mask.extent(1), NZ = mask.extent(2);
# ~ mass = 0;
# ~ for (int i1=0; i1<NX; ++i1)
# ~ for (int i2=0; i2<NY; ++i2)
# ~ for (int i3=0; i3<NZ; ++i3)
# ~ if ( mask(i1,i2,i3)>=0 )
# ~ mass( mask(i1,i2,i3) ) += density(i1,i2,i3);
# ~ '''
# ~
# ~ # Call the C++ code to compute the mass
# ~ noObjects = mask.max() + 1
# ~ mass = np.zeros( noObjects, np.float32 )
# ~ inline( MMFObjectsMassCode, ['mask','density', 'mass'], type_converters=converters.blitz )
# ~ spacing = np.zeros(3,np.float32)
# ~ spacing[:] = boxLength[:] / mask.shape[:]
# ~ cellVolume =  spacing[0] * spacing[1] * spacing[2]
# ~ mass *= cellVolume
# ~
# ~ return mass


# ~ def MMFObjectsCenter(mask,density,boxLength,VERBOSE=True):
# ~ """ Finds the CM of the MMF objects. Usefull for finding the CM of the nodes, less usefull for the other features.
# ~ mask = an integer numpy array giving the label of the MMF object (-1=no object, 0..n=the label of the objects)
# ~ density = a float numpy array giving the density in each cell of the mask
# ~ boxLength = the box length along each dimension
# ~ """
# ~
# ~ if VERBOSE: print "Computing the center of mass for the MMF objects ..."
# ~ # Use C++ to get the CM position
# ~ MMFObjectsCenterCode = '''
# ~ #line 1000 "MMFObjectsCenterCode"
# ~ int const NX = mask.extent(0), NY = mask.extent(1), NZ = mask.extent(2);
# ~ blitz::Array<float,1> objectVolume( objectCenter.extent(0) );
# ~ objectVolume = float(0.);
# ~ objectCenter = float(0.);
# ~ for (int i1=0; i1<NX; ++i1)
# ~ for (int i2=0; i2<NY; ++i2)
# ~ for (int i3=0; i3<NZ; ++i3)
# ~ if ( mask(i1,i2,i3)>=0 )
# ~ {
# ~ int const obj = mask(i1,i2,i3);
# ~ objectVolume(obj) += density(i1,i2,i3);
# ~ objectCenter(obj,0) += density(i1,i2,i3) * i1;
# ~ objectCenter(obj,1) += density(i1,i2,i3) * i2;
# ~ objectCenter(obj,2) += density(i1,i2,i3) * i3;
# ~ }
# ~
# ~ for (int i=0; i<objectCenter.extent(0); ++i)
# ~ for (int j=0; j<3; ++j)
# ~ objectCenter(i,j) /= objectVolume(i);
# ~ '''
# ~
# ~ # Call the C++ code to compute the object CM
# ~ if mask.shape!=density.shape:
# ~ throwError( "The 'mask' and 'density' arguments of function 'MMFObjectsCenter' must be 3D array having the same shape. The 'mask' shape = %s while the 'density' shape = %s." % (str(mask.shape),str(density.shape)) )
# ~ if mask.ndim!=3:
# ~ throwError( "The 'mask' and 'density' arguments of function 'MMFObjectsCenter' must be 3D numpy arrays. But the 'mask' and 'density' arguments are %iD arrays." % mask.ndim )
# ~ noObjects = mask.max() + 1
# ~ objectCenter = np.zeros( (noObjects,3), np.float32 )
# ~ inline( MMFObjectsCenterCode, ['mask','density', 'objectCenter'], type_converters=converters.blitz )
# ~ spacing = np.zeros(3,np.float32)
# ~ spacing[:] = boxLength[:] / mask.shape[:]
# ~ objectCenter[:,:] *= spacing[:]
# ~
# ~ return objectCenter


# ~ def matchObjectsAccordingToCenter(objectCenter1,objectCenter2,tolerance,boxLength,VERBOSE=True):
# ~ """ The function tries to match objects from two different analysis using the best match of their centers.
# ~ objectCenter1 = a 2d numpy array giving the first objects center
# ~ objectCenter2 = a 2d numpy array giving the second objects center
# ~ tolerance = the largest distance between two cenetrs at which two objects are still the same
# ~ boxLength = the length of the periodic box (3 entries)
# ~ This function finds the corresponding object using the object with the closest mass/size to the matching object (once it finds a match, does not continue the search at objects of lower mass).
# ~ """
# ~
# ~ if VERBOSE: print "Matching objects between two different analysis using the object centers ... "
# ~ # Use C++ code to make the matching
# ~ matchObjectsCentersSupportCode = '''
# ~ #line 1000 "matchObjectsCentersSupportCode"
# ~ template <typename T>
# ~ T periodicDistance(T *pos1, T *pos2, T *length)
# ~ {
# ~ T temp, dist = 0.;
# ~ for (int i=0; i<3; ++i)
# ~ {
# ~ temp = pos1[i] - pos2[i];
# ~ if ( std::fabs(temp)>std::fabs(temp+length[i]) ) temp += length[i];
# ~ else if ( std::fabs(temp)>std::fabs(temp-length[i]) ) temp -= length[i];
# ~ dist += temp*temp;
# ~ }
# ~ return dist;
# ~ }
# ~ '''
# ~ matchObjectsCentersCode = '''
# ~ #line 1000 "matchObjectsCentersCode"
# ~ float tol2 = float(tolerance) * float(tolerance);
# ~ float length[] = { boxLength(0), boxLength(1), boxLength(2) };
# ~ int const N1 = objectCenter1.extent(0), N2 = objectCenter2.extent(0);
# ~ blitz::Array<bool,1> matched( N2 );
# ~ matched = false;
# ~ int start = 0;
# ~ for (int i=0; i<N1; ++i)
# ~ {
# ~ float pos1[] = { objectCenter1(i,0), objectCenter1(i,1), objectCenter1(i,2) };
# ~ for (int j=start; j<N2; ++j)    // all objects before 'start' have been matched
# ~ {
# ~ float pos2[] = { objectCenter2(j,0), objectCenter2(j,1), objectCenter2(j,2) };
# ~ float distance = periodicDistance(pos1,pos2,length);
# ~ if ( distance<tol2 and not matched(j) )
# ~ {
# ~ match(i) = j;
# ~ centerDistance(i) = distance;
# ~ matched(j) = true;
# ~ break;
# ~ }
# ~ }
# ~ while( matched(start+1) )   // if next object is matched, increase the value of 'start'
# ~ {
# ~ ++start;
# ~ }
# ~ }
# ~ '''
# ~
# ~ # Call the C++ code
# ~ boxLength = boxLength.astype( np.float32 )
# ~ match = np.zeros( objectCenter1.shape[0], np.int32 )
# ~ centerDistance = np.zeros( objectCenter1.shape[0], np.float32 )
# ~ match[:] = -1
# ~ inline( matchObjectsCentersCode, ['objectCenter1','objectCenter2', 'tolerance', 'match', 'centerDistance', 'boxLength'], type_converters=converters.blitz, support_code=matchObjectsCentersSupportCode )
# ~
# ~ return (match, centerDistance)


# ~ def MMFFilamentSplit(objects,response,eigenVectors,radius,boxLength,VERBOSE=True):
# ~ """This code splits a given filamentary array according to the filament directions at a given filament intersection.
# ~ """
# ~ funcName = "MMFFilamentSplit"
# ~ checkArraySizes( [objects.shape,response.shape,eigenVectors.shape[0:3]], ["objects","response","EigenVectors"], funcName )
# ~ if VERBOSE: print "Spliting the filamentary network to individual filaments on a %s size map ..." % str(objects.shape)
# ~
# ~ # the C++ code that does the actual computation
# ~ MMFFilamentSplitSupportCode = """ #include "%s/MMF_computations.cc" """ % currentDirectory
# ~
# ~ MMFFilamentSplitCode = """
# ~ #line 1000 "MMFFilamentSplitCode"
# ~ Real radius = Real( parameters(0) );
# ~ grid[0] = objects.extent(0); grid[1] = objects.extent(1), grid[2] =  objects.extent(2);
# ~ int indices[NO_DIM];
# ~ blitz::Array<int,2> offsets(10,NO_DIM);
# ~ int const noNeighbors = cellsOffsets( radius,  offsets ); // returns the neighbor up to radius 'radius' around the grid cell at origin
# ~ blitz::Array<int,2> cells(noNeighbors,NO_DIM);  // array to keep track of the neighbors indices for each grid cell
# ~
# ~ // loop over all the valid cells
# ~ int i=0;
# ~ for (int i1=0; i1<objects.extent(0); ++i1)
# ~ {
# ~ indices[0] = i1;
# ~ for (int i2=0; i2<objects.extent(1); ++i2)
# ~ {
# ~ indices[1] = i2;
# ~ for (int i3=0; i3<objects.extent(2); ++i3)
# ~ {
# ~ if ( objects(i1,i2,i3)<0 ) continue;
# ~ indices[2] = i3;
# ~ int noCells = cellsWithinInfluenceSphere( indices, grid, offsets, cells, objects(i1,i2,i3), objects );
# ~ coherence(i1,i2,i3) = directionalCoherence( indices, cells, noCells, eigenVectors );
# ~ }
# ~ }
# ~ }
# ~ """
# ~
# ~ # call the C++ code that does the computation
# ~ parameters = np.zeros( 1, np.float32 )
# ~ parameters[0] = radius / (boxLength[0]/objects.shape[0])
# ~ coherence = np.zeros( objects.shape, np.float32 )
# ~ inline( MMFFilamentSplitCode, ['parameters','objects','response', 'eigenVectors', 'coherence'], type_converters=converters.blitz, support_code=MMFFilamentSplitSupportCode )
# ~
# ~ return coherence


# ~ def MMFFeatureProperties(Mask,Directions,Density,Feature,Radius=1.,BoxLength=1.):
# ~ """Computes the properties associated to the thickness and mass density of filaments and walls (feature=3 and feature=2 respectively). """
# ~ funcName = 'MMFFeatureProperties'
# ~ listName = {3:'filaments',2:'walls'}
# ~ print "Computing the properties of %s using contraction to a line/plane for filaments and walls ..." % listName[Feature]
# ~
# ~ if Feature not in [2,3]:
# ~ throwError( "The input argument 'Feature' to function '%s' must have one of the values %s" % (funcName,str([2,3])) )
# ~ checkArraySizes( [Mask.shape,Density.shape], ["Mask","Density"], funcName )
# ~ boxLength = np.array(3,np.float32)
# ~ boxLength = BoxLength
# ~
# ~
# ~ # use the C++ codes to compute the quantities
# ~ MMFFeaturePropertiesSupportCode = """ #include "%s/MMF_computations.cc" """ % currentDirectory
# ~
# ~ MMFFeaturePropertiesCode = """
# ~ #line 1000 "MMFFeaturePropertiesCode"
# ~
# ~ // sets the global variables 'grid' and 'boxLength'
# ~ setParameters( mask.extent(0), mask.extent(1), mask.extent(2), parameters(0), parameters(1), parameters(2) );
# ~ int noNeighbors = int( parameters(5) );
# ~
# ~ //get the unique MMF objects
# ~ blitz::Array<int,1> tempObjectSize(2);
# ~ identifyDistinctObjects( mask, noNeighbors, &tempObjectSize );
# ~
# ~ // call the function that computes the values
# ~ MMFFeatureProperties( mask, Density, Directions, int(parameters(3)), parameters(4), sizeData, massData, averageData );
# ~ """
# ~
# ~
# ~ # Call the C++ code
# ~ mask = MMFMask(Mask)
# ~ parameters = np.zeros( 6, np.float32 )
# ~ parameters[0:3] = BoxLength
# ~ parameters[3], parameters[4], parameters[5] = Feature, Radius, 6    # the last is the number of neighbors
# ~ sizeData = np.zeros( (101,2), np.float32 )
# ~ sizeData[:,0] = analysis.binBoundaries( (0.,10.),100,bin_type='linear')
# ~ massData = np.zeros( (71,2), np.float32 )
# ~ massData[:,0] = analysis.binBoundaries( (1.e9,1.e15),70,bin_type='logarithm')
# ~ averageData = np.zeros( 10, np.float32 )
# ~
# ~ inline( eval(funcName+'Code'), ['mask','Directions','Density','parameters','sizeData','massData','averageData'], type_converters=converters.blitz, support_code=eval(funcName+'SupportCode'), libraries=['gsl','gslcblas', 'm'], extra_compile_args=['-O3 -fopenmp -DENABLE_OPENMP'],extra_link_args=['-lgomp'], force=0 )
# ~
# ~ sizeStep, massStep = sizeData[1,0]-sizeData[0,0], np.log10(massData[1,0]/massData[0,0])
# ~ sizeData[:-1,0] = (sizeData[:-1,0] + sizeData[1:,0] ) /2.
# ~ sizeData = np.resize( sizeData, (sizeData.shape[0]-1,sizeData.shape[1]) )
# ~ sizeData[:,1] /= sizeStep
# ~ massData[:-1,0] = np.sqrt(massData[:-1,0] * massData[1:,0] )
# ~ massData = np.resize( massData, (massData.shape[0]-1,massData.shape[1]) )
# ~ massData[:,1] /= massStep
# ~
# ~ return (sizeData,massData,averageData)


# ~ def MMFFeatureDirection(Mask,Feature,Radius=1.,BoxLength=1.):
# ~ """Returns the inner axis of filaments and the plane of walls. It returns the positions to which the object grid cells have been contracted to."""
# ~ funcName = 'MMFFeatureDirection'
# ~ listName = {3:'filaments',2:'walls'}
# ~ print "Computing the axis/plane of %s using contraction to a line/plane for filaments and walls ..." % listName[Feature]
# ~
# ~ if Feature not in [2,3]:
# ~ throwError( "The input argument 'Feature' to function '%s' must have one of the values %s" % (funcName,str([2,3])) )
# ~ boxLength = np.array(3,np.float32)
# ~ boxLength = BoxLength
# ~
# ~ # use the C++ codes to compute the quantities
# ~ MMFFeatureDirectionSupportCode = """ #include "%s/MMF_computations.cc" """ % currentDirectory
# ~
# ~ MMFFeatureDirectionCode = """
# ~ #line 1000 "%sCode"
# ~
# ~ // sets the global variables 'grid' and 'boxLength'
# ~ setParameters( mask.extent(0), mask.extent(1), mask.extent(2), parameters(0), parameters(1), parameters(2) );
# ~ int noNeighbors = int( parameters(5) );
# ~
# ~ //get the unique MMF objects
# ~ blitz::Array<int,1> tempObjectSize(2);
# ~ identifyDistinctObjects( mask, noNeighbors, &tempObjectSize );
# ~
# ~ // compute the spin of the filaments and plane of the wall
# ~ int noCells = int( parameters(6) );
# ~ blitz::Array<int,2> cellIndices(noCells,NO_DIM);
# ~ blitz::Array<Real,2> iSpinePosition(noCells,NO_DIM);
# ~ MMFObjectDirection( mask, int(parameters(3)), parameters(4), cellIndices, iSpinePosition, directionVectors );
# ~ """ % funcName
# ~
# ~ # Call the C++ code
# ~ noCells = (Mask!=0).sum()
# ~ mask = MMFMask(Mask)
# ~ parameters = np.zeros( 7, np.float32 )
# ~ parameters[0:3] = BoxLength
# ~ parameters[3], parameters[4], parameters[5], parameters[6] = Feature, Radius, 6, noCells    # 'the value 6'= the number of neighbors
# ~ directionVectors = np.zeros( (noCells,3), np.float32 )     # direction associated to each valid filament/wall point
# ~ inline( eval(funcName+'Code'), ['mask','directionVectors','parameters'], type_converters=converters.blitz, support_code=eval(funcName+'SupportCode'), libraries=['gsl','gslcblas', 'm'], extra_compile_args=['-O3 -fopenmp -DENABLE_OPENMP'],extra_link_args=['-lgomp'], force=0 ) #include_dirs=includePaths, library_dirs=libraryPaths,
# ~
# ~ return directionVectors


# ~ def MMFFeatureSpine(Mask,Response,EigenVectors,Feature,Radius=1.,BoxLength=1.):
# ~ """Returns the inner axis of filaments and the plane of walls. It returns the positions to which the object grid cells have been contracted to."""
# ~ funcName = 'MMFFeatureProperties'
# ~ listName = {3:'filaments',2:'walls'}
# ~ print "Computing the axis/plane of %s using contraction to a line/plane for filaments and walls ..." % listName[Feature]
# ~
# ~ if Feature not in [2,3]:
# ~ throwError( "The input argument 'Feature' to function '%s' must have one of the values %s" % (funcName,str([2,3])) )
# ~ checkArraySizes( [Mask.shape,Response.shape,EigenVectors.shape[0:3]], ["Mask","Response","EigenVectors"], funcName )
# ~ boxLength = np.array(3,np.float32)
# ~ boxLength = BoxLength
# ~
# ~ # use the C++ codes to compute the quantities
# ~ MMFFeatureSpineSupportCode = """ #include "%s/MMF_computations.cc" """ % currentDirectory
# ~
# ~ MMFFeatureSpineCode = """
# ~ #line 1000 "MMFFeatureSpineCode"
# ~
# ~ // sets the global variables 'grid' and 'boxLength'
# ~ setParameters( mask.extent(0), mask.extent(1), mask.extent(2), parameters(0), parameters(1), parameters(2) );
# ~ int noNeighbors = int( parameters(5) );
# ~
# ~ //get the unique MMF objects
# ~ blitz::Array<int,1> tempObjectSize(2);
# ~ identifyDistinctObjects( mask, noNeighbors, &tempObjectSize );
# ~
# ~ // compute the spin of the filaments and plane of the wall
# ~ int noCells = int( parameters(6) );
# ~ blitz::Array<int,2> cellIndices(noCells,NO_DIM);
# ~ blitz::Array<Real,2> iSpinePosition(noCells,NO_DIM);
# ~ blitz::Array<Real,2> directionVectors(noCells,NO_DIM);
# ~ MMFObjectDirection( mask, parameters(3), parameters(4), cellIndices, iSpinePosition, directionVectors );
# ~ MMFContraction( iSpinePosition, directionVectors, parameters(3), parameters(4) );
# ~ std::cout << "Done\\n";
# ~
# ~ //copy the results to the output array
# ~ for (int i=0; i<noCells; ++i)
# ~ for (int j=0; j<NO_DIM; ++j)
# ~ {
# ~ spinePosition(i,j) = cellIndices(i,j) * dx[j] + dx[j]/2;
# ~ spinePosition(i,j+3) = iSpinePosition(i,j);
# ~ spinePosition(i,j+6) = directionVectors(i,j);
# ~ spinePosition(i,j+9) = EigenVectors(cellIndices(i,0),cellIndices(i,1),cellIndices(i,2),j);
# ~ }
# ~ """
# ~
# ~ # Call the C++ code
# ~ noCells = (Mask!=0).sum()
# ~ mask = MMFMask(Mask)
# ~ parameters = np.zeros( 7, np.float32 )
# ~ parameters[0:3] = BoxLength
# ~ parameters[3], parameters[4], parameters[5], parameters[6] = Feature, Radius, 6, noCells    # 'the value 6'= the number of neighbors
# ~ spinePosition = np.zeros( (noCells,12), np.float32 )     #first 3 entries: initial position of the cell, last 3 entries: spine position of the cell
# ~ inline( MMFFeatureSpineCode, ['mask','Response','EigenVectors','parameters','spinePosition'], type_converters=converters.blitz, support_code=MMFFeatureSpineSupportCode, libraries=['gsl','gslcblas', 'm'], extra_compile_args=['-O3'], force=0 )
# ~
# ~ return spinePosition


def NEXUSCombineEnvironments(nodeCleanData, filaCleanData, wallCleanData, VERBOSE=True):
    """Combines the node, filament and wall clean responses into a single combined data structure where 0=field, 2=walls, 3=filaments and 4=nodes."""
    if VERBOSE:
        print(
            "Combining the node, filament and wall environments into a single data structure ..."
        )
    result = wallCleanData.copy()
    result[wallCleanData == 1] = 2
    result[filaCleanData == 1] = 3
    result[nodeCleanData == 1] = 4
    return result


def NEXUSEnvironmentProperties(
    allCleanData, propertiesData, env, masking=(True, True, True), VERBOSE=True
):
    """Creates a properties file on the initial data grid for the given environment."""
    funcName = "NEXUSEnvironmentProperties"
    envList = ["fila", "wall", 3, 2]
    if env not in envList:
        throwError(
            "Invalid value for the 'env' argument in the '%s' function. Allowed values for this argument are: %s"
            % (funcName, str(envList))
        )
    elif env in ["fila", "wall"]:
        env = [3, 2][int(env == "wall")]
    envName = ["filament", "wall"][int(env == 2)]
    if propertiesData.ndim <= 1:
        throwError(
            "The argument 'propertiesData. in the '%s' function needs to be at least a 2D numpy array. The last dimension gives the number of properties for each grid cell."
        )

    # get the mask and check compatibility with the input data
    if VERBOSE:
        print("Getting the %s properties on a grid cell data format ..." % envName)
    select = allCleanData == env
    if masking[0]:
        select += allCleanData == 4
        maskText = "node"
    if masking[1] and env == 2:
        select += allCleanData == 3
        if maskText == "":
            maskText = "filament"
        else:
            maskText = "%s + filament" % maskText
    noElements, noDatas = np.sum(select), propertiesData.size / propertiesData.shape[-1]
    if noElements != noDatas:
        throwError(
            "The number of valid environment grid cells and the size of the properties data does not match. There are %i valid environment grid cells while there are %i properties data. The requested environment is '%s' using %s masks."
            % (noElements, noDatas, envName, maskText)
        )

    # copy the results to grid
    result = np.zeros((allCleanData.size, propertiesData.shape[-1]), np.float32)
    result[select.flatten(), :] = propertiesData
    return result
