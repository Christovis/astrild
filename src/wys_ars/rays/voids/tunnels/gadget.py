import os.path
import numpy as np
from wys_ars.rays.voids.tunnels.miscellaneous import (
    throwError,
    throwWarning,
    charToString,
)


GadgetFileType = {1: "Gadget1", 2: "Gadget2", 3: "HDF5", -1: "Unknown"}


class GadgetParticles:
    """ Class used for storing and accessing the data stored in a Gadget file. It keeps track of the properties of Dark Matter particles: position, velocity, identity and mass. """

    def __init__(self):
        self.noParticles = np.uint64(0)
        self.hasHeader = False
        self.hasPos = False
        self.hasVel = False
        self.hasIds = False
        self.hasMass = False

    def AddHeader(self, Header):
        self.header = Header
        self.hasHeader = True

    def AddPos(self, positions):
        self.pos = positions
        self.pos.shape = (-1, 3)
        self.hasPos = True
        self.UpdateNoParticles(self.pos.size / 3, "positions")

    def AddVel(self, velocities):
        self.vel = velocities
        self.vel.shape = (-1, 3)
        self.hasVel = True
        self.UpdateNoParticles(self.vel.size / 3, "velocities")

    def AddIds(self, ids):
        self.ids = ids
        self.ids.shape = -1
        self.hasIds = True
        self.UpdateNoParticles(self.ids.size, "identities")

    def AddMass(self, masses):
        self.mass = masses
        if self.mass is not None:
            self.mass.shape = -1
            self.hasMass = True
            self.UpdateNoParticles(self.mass.size, "masses")

    def Header(self):
        if self.hasHeader:
            return self.header
        else:
            throwError(
                "No header was added to class 'GadgetParticles' via the function 'AddHeader'. Hence you cannot use the function 'Header' to access the Gadget header."
            )

    def Pos(self):
        if self.hasPos:
            return self.pos
        else:
            throwError(
                "No position array was added to class 'GadgetParticles' via the function 'AddPos'. Hence you cannot use the function 'Pos' to access the position array."
            )

    def Vel(self):
        if self.hasVel:
            return self.vel
        else:
            throwError(
                "No velocity array was added to class 'GadgetParticles' via the function 'AddVel'. Hence you cannot use the function 'Vel' to access the velocities array."
            )

    def Ids(self):
        if self.hasIds:
            return self.ids
        else:
            throwError(
                "No particle identities array was added to class 'GadgetParticles' via the function 'AddIds'. Hence you cannot use the function 'Ids' to access the position array."
            )

    def Mass(self):
        if self.hasMass:
            return self.mass
        elif self.hasHeader and self.header.mass[1] != 0.0:
            return np.array([self.header.mass[1]], "f4")
        else:
            throwError(
                "No particle mass array was added to class 'GadgetParticles' via the function 'AddMass'. Hence you cannot use the function 'Mass' to access the position array."
            )

    def VariableMass(self):
        if self.hasMass:
            return True
        else:
            return False

    def Update(self, selection):
        """Update the particle information using only the particles selected via 'selection'"""
        if selection.size != self.noParticles:
            throwError(
                "In function 'GadgetParticles::Update'. You are trying to update the particle selection using a numpy array that has a different size than the number of particles in class 'GadgetParticles'. Number of particles = %i while the selection array has length = %i."
                % (self.noParticles, selection.size)
            )
        self.noParticles = np.sum(selection)
        if self.hasHeader:
            self.header.npart[1] = self.noParticles
            self.header.npartTotal[1] = self.noParticles
        self.pos = self.pos[selection, :]
        self.vel = self.vel[selection, :]
        self.ids = self.ids[selection]
        if self.VariableMass():
            self.mass = self.mass[selection]

    def UpdateNoParticles(self, numberPart, arrayName):
        if numberPart != self.noParticles and self.noParticles != 0:
            throwWarning(
                "The Gadget particle array '%s' has a different number of particles than the previous array/arrays."
                % arrayName
            )
        if self.noParticles == 0:
            self.noParticles = numberPart

    def CheckDataCompletness(
        self, HEADER=True, POS=True, VEL=True, ID=True, MASS=True
    ):
        completness = True
        if HEADER and not self.hasHeader:
            throwWarning(
                "'GadgetParticles' completness check failled: Gadget header is missing. The Gadget header must be added to the class using the function 'AddHeader(header)'."
            )
            completness = False
        if POS and not self.hasPos:
            throwWarning(
                "'GadgetParticles' completness check failled: Gadget particle position array is missing. The position array must be added to the class using the function 'AddPos(position)'."
            )
            completness = False
        if VEL and not self.hasVel:
            throwWarning(
                "'GadgetParticles' completness check failled: Gadget particle velocity array is missing. The velocity array must be added to the class using the function 'AddVel(velocity)'."
            )
            completness = False
        if ID and not self.hasIds:
            throwWarning(
                "'GadgetParticles' completness check failled: Gadget particle id array is missing. The id array must be added to the class using the function 'AddIds(ids)'."
            )
            completness = False
        if MASS and not self.hasMass and self.Header().mass[1] == 0.0:
            throwWarning(
                "'GadgetParticles' completness check failled: Gadget particle mass array is missing (and there is no particle mass in the Gadget header). The masses array must be added to the class using the function 'AddMass(masses)'."
            )
            completness = False
        return completness

    def SameNumberOfParticles(self, POS=True, VEL=True, ID=True, MASS=True):
        noParticles = 0
        sameNumber = True
        self.CheckDataCompletness(
            HEADER=False, POS=POS, VEL=VEL, ID=ID, MASS=MASS
        )
        if POS and self.hasPos:
            if noParticles == 0:
                noParticles = self.Pos().size / 3
            elif noParticles != self.Pos().size / 3:
                sameNumber = False
        if VEL and self.hasVel:
            if noParticles == 0:
                noParticles = self.Vel().size / 3
            elif noParticles != self.Vel().size / 3:
                sameNumber = False
        if ID and self.hasIds:
            if noParticles == 0:
                noParticles = self.Ids().size
            elif noParticles != self.Ids().size:
                sameNumber = False
        if MASS and self.hasMass:
            if noParticles == 0:
                noParticles = self.Mass().size
            elif noParticles != self.Mass().size:
                sameNumber = False
        return (noParticles, sameNumber)


class GadgetHeader:
    """ A class used for reading and storing the header of a Gadget snapshot file. It uses the numpy class to define the variables. """

    headerSize = 256
    fillSize = headerSize - 20 * 8 - 12 * 4

    def __init__(self):
        self.npart = np.zeros(6, dtype=np.uint32)
        self.mass = np.zeros(6, dtype=np.float64)
        self.time = np.float64(0.0)
        self.redshift = np.float64(0.0)
        self.flag_sfr = np.int32(0)
        self.flag_feedback = np.int32(0)
        self.npartTotal = np.zeros(6, dtype=np.uint32)
        self.flag_cooling = np.int32(0)
        self.num_files = np.int32(1)
        self.BoxSize = np.float64(0.0)
        self.Omega0 = np.float64(0.0)
        self.OmegaLambda = np.float64(0.0)
        self.HubbleParam = np.float64(0.0)

        self.flag_stellarage = np.int32(0)
        self.flag_metals = np.int32(0)
        self.num_total_particles_hw = np.zeros(6, dtype=np.uint32)
        self.flag_entropy_instead_u = np.int32(0)
        self.flag_doubleprecision = np.int32(0)
        self.flag_ic_info = np.int32(0)
        self.lpt_scalingfactor = np.float32(0.0)

        self.fill = np.zeros(GadgetHeader.fillSize, dtype="c")

    def SetType(self):
        self.time = np.float64(self.time)
        self.redshift = np.float64(self.redshift)
        self.flag_sfr = np.int32(self.flag_sfr)
        self.flag_feedback = np.int32(self.flag_feedback)
        self.flag_cooling = np.int32(self.flag_cooling)
        self.num_files = np.int32(self.num_files)
        self.BoxSize = np.float64(self.BoxSize)
        self.Omega0 = np.float64(self.Omega0)
        self.OmegaLambda = np.float64(self.OmegaLambda)
        self.HubbleParam = np.float64(self.HubbleParam)
        self.flag_stellarage = np.int32(self.flag_stellarage)
        self.flag_metals = np.int32(self.flag_metals)
        self.flag_entropy_instead_u = np.int32(self.flag_entropy_instead_u)
        self.flag_doubleprecision = np.int32(self.flag_doubleprecision)
        self.flag_ic_info = np.int32(self.flag_ic_info)
        self.lpt_scalingfactor = np.float32(self.lpt_scalingfactor)

    def nbytes(self):
        __size = (
            self.npart.nbytes
            + self.mass.nbytes
            + self.time.nbytes
            + self.redshift.nbytes
            + self.flag_sfr.nbytes
            + self.flag_feedback.nbytes
            + self.npartTotal.nbytes
            + self.flag_cooling.nbytes
            + self.num_files.nbytes
            + self.BoxSize.nbytes
            + self.Omega0.nbytes
            + self.OmegaLambda.nbytes
            + self.HubbleParam.nbytes
            + self.flag_stellarage.nbytes
            + self.flag_metals.nbytes
            + self.num_total_particles_hw.nbytes
            + self.flag_entropy_instead_u.nbytes
            + self.flag_doubleprecision.nbytes
            + self.flag_ic_info.nbytes
            + self.lpt_scalingfactor.nbytes
            + self.fill.nbytes
        )
        return __size

    def dtype(self):
        __dt = np.dtype(
            [
                ("npart", np.uint32, 6),
                ("mass", np.float64, 6),
                ("time", np.float64),
                ("redshift", np.float64),
                ("flag_sfr", np.int32),
                ("flag_feedback", np.int32),
                ("npartTotal", np.uint32, 6),
                ("flag_cooling", np.int32),
                ("num_files", np.int32),
                ("BoxSize", np.float64),
                ("Omega0", np.float64),
                ("OmegaLambda", np.float64),
                ("HubbleParam", np.float64),
                ("flag_stellarage", np.int32),
                ("flag_metals", np.int32),
                ("num_total_particles_hw", np.uint32, 6),
                ("flag_entropy_instead_u", np.int32),
                ("flag_doubleprecision", np.int32),
                ("flag_ic_info", np.int32),
                ("lpt_scalingfactor", np.float32),
                ("fill", "c", GadgetHeader.fillSize),
            ]
        )
        return __dt

    def TupleAsString(self):
        return "( self.npart, self.mass, self.time, self.redshift, self.flag_sfr, self.flag_feedback, self.npartTotal, self.flag_cooling, self.num_files, self.BoxSize, self.Omega0, self.OmegaLambda, self.HubbleParam, self.flag_stellarage, self.flag_metals, self.num_total_particles_hw, self.flag_entropy_instead_u, self.flag_doubleprecision, self.flag_ic_info, self.lpt_scalingfactor, self.fill )"

    def Tuple(self):
        return eval(self.TupleAsString())

    def fromfile(
        self, f, BUFFER=True, bufferType=np.dtype("i4"), switchEndian=False
    ):
        if BUFFER:
            __buffer1 = np.fromfile(f, bufferType, 1)[0]
        A = np.fromfile(f, self.dtype(), 1)[0]
        if switchEndian:
            for i in range(len(A)):
                if A[i].ndim >= 1:
                    A[i][:] = A[i].byteswap()
                else:
                    A[i] = A[i].byteswap()
        if BUFFER:
            __buffer2 = np.fromfile(f, bufferType, 1)[0]
            if switchEndian:
                __buffer1 = __buffer1.byteswap()
                __buffer2 = __buffer2.byteswap()
            if __buffer1 != __buffer2 or __buffer1 != GadgetHeader.headerSize:
                throwError(
                    "Error reading the header of the Gadget file. 'buffer1'=%i while 'buffer2'=%i when both should be %i."
                    % (__buffer1, __buffer2, GadgetHeader.headerSize)
                )
        exec("%s = A" % self.TupleAsString())

    def tofile(self, f, BUFFER=True, bufferType=np.dtype("i4")):
        self.SetType()
        __A = np.array([self.Tuple()], dtype=self.dtype())
        __buffer = np.array([__A.nbytes], dtype=bufferType)
        if __A.nbytes != GadgetHeader.headerSize:
            throwError(
                "When writing the Gadget snapshot header to file. The header size is %i which is different from the expected size of %i."
                % (__A.nbytes, GadgetHeader.headerSize)
            )
        if BUFFER:
            __buffer.tofile(f)
        __A.tofile(f)
        if BUFFER:
            __buffer.tofile(f)

    def PrintValues(self):
        print("The values contained in the Gadget header:")
        print("  npart        = ", self.npart)
        print("  mass         = ", self.mass)
        print("  time         = ", self.time)
        print("  redshift     = ", self.redshift)
        print("  npartTotal   = ", self.npartTotal)
        print("  num_files    = ", self.num_files)
        print("  BoxSize      = ", self.BoxSize)
        print("  Omega0       = ", self.Omega0)
        print("  OmegaLambda  = ", self.OmegaLambda)
        print("  HubbleParam  = ", self.HubbleParam)
        print("  fill         = ", charToString(self.fill))
        print()

    def Description(self):
        __description = [
            (
                "npartTotal",
                "=%s - the total number of particles in the given Gadget snapshot"
                % " ".join(str(self.npartTotal).split()),
            ),
            (
                "mass",
                "=%s - the mass of each particle in the Gadget snapshot (10^10 M_0/h)"
                % " ".join(str(self.mass).split()),
            ),
            ("time", "=%f - the scaling factor of the snapshot" % self.time),
            ("redshift", "=%f - the redshift of the snapshot" % self.redshift),
            (
                "BoxSize",
                "=%f - the size of the simulation box in comoving corodinates (kpc/h)"
                % self.BoxSize,
            ),
            ("Omega0", "=%f - the matter density" % self.Omega0),
            (
                "OmegaLambda",
                "=%f - the Lambda energy density" % self.OmegaLambda,
            ),
            (
                "HubbleParam",
                "=%f - Hubble parameter 'h' where the Hubble constant H=100 km/s h"
                % self.HubbleParam,
            ),
            (
                "gadgetFill",
                "='%s' - additional information" % charToString(self.fill),
            ),
        ]
        return __description

    def AddProgramCommands(self, commands):
        """Adds the program options used to obtain the current results to the 'fill' array in the header."""
        newCommands = self.fill.tostring().rstrip("\x00") + commands + " ;  "
        choice = int(len(newCommands) < GadgetHeader.fillSize)
        newLen = [GadgetHeader.fillSize, len(newCommands)][choice]
        newOff = [len(newCommands) - GadgetHeader.fillSize, 0][choice]
        self.fill[:newLen] = newCommands[newOff:newLen]


def getGadgetFileType(fileName, bufferType=np.dtype("i4")):
    """Retuns the type of gadget file and other properties related to the data format."""
    gadgetFileType = GadgetFileType[1]
    switchEndian = False

    # read the 1st 4 bites and determine the type of the gadget file
    entry = np.fromfile(fileName, bufferType, 1)
    if entry[0] == 8:
        gadgetFileType = GadgetFileType[2]
    elif entry[0] != 256:  # try to switch endian type
        entry2 = entry.byteswap()[0]
        switchEndian = True
        if entry2 == 8:
            gadgetFileType = GadgetFileType[2]
        elif entry2 != 256:  # test for HDF5 file
            gadgetFileType = GadgetFileType[-1]

    offsetSize = 0
    if gadgetFileType == GadgetFileType[2]:
        offsetSize = 16  # jump 16 bytes when reading Gadget2 snapshot files

    return gadgetFileType, switchEndian, offsetSize


def gadgetMultipleFiles(rootName, fileIndex):
    """ Returns the name of gadget file 'fileIndex' when a snapshot is saved in multiple binary files.
    It takes 2 arguments: root name of the files and the file number whose name is requested (from 0 to GadgetHeader.num_files-1)."""
    return rootName + "%i" % fileIndex


def readArrayEntries(array, file, startPosition, noEntries):
    """Reads from file 'file' 'noEntries' entries into the array 'array'. The entries are written starting at position 'startPosition' as in a flattened array."""
    maxChunckOfData = 1024 ** 3
    noChuncks = noEntries / maxChunckOfData
    for i in range(noChuncks):
        array[startPosition : startPosition + maxChunckOfData] = np.fromfile(
            file, array.dtype, maxChunckOfData
        )
        startPosition += maxChunckOfData
    noLeftOver = noEntries - maxChunckOfData * noChuncks
    array[startPosition : startPosition + noLeftOver] = np.fromfile(
        file, array.dtype, noLeftOver
    )
    return array


def gadgetTotalParticleCount(fileRoot, noFiles, VERBOSE=True):
    """Computes the total number of particles found in the file."""
    output = np.zeros(6, np.int64)
    for i in range(noFiles):
        h = readGadgetHeader(fileRoot, INDEX=i, VERBOSE=False)
        output[:] += h.npart[:]
    if VERBOSE:
        print("Total number of particles: ", output)
    return output


def gadgetDataType(
    file,
    noParticles,
    hasMass,
    offsetSize,
    switchEndian,
    bufferType=np.int32,
    VERBOSE=True,
):
    """Finds what is the type of the gadget data."""
    floatTypes = {4: np.float32, 8: np.float64}
    intTypes = {4: np.int32, 8: np.int64}
    f = open(file, "rb")

    # read header
    f.seek(offsetSize, 1)
    buffer1 = np.fromfile(f, bufferType, 1)[0]
    f.seek(buffer1 + buffer1.nbytes, 1)

    # read positions block
    f.seek(offsetSize, 1)
    buffer1 = np.fromfile(f, bufferType, 1)[0]
    f.seek(buffer1 + buffer1.nbytes, 1)
    posSize = buffer1 / noParticles / 3
    posType = floatTypes[posSize]

    # read velocity block
    f.seek(offsetSize, 1)
    buffer1 = np.fromfile(f, bufferType, 1)[0]
    f.seek(buffer1 + buffer1.nbytes, 1)
    velSize = buffer1 / noParticles / 3
    velType = floatTypes[velSize]

    # read id block
    f.seek(offsetSize, 1)
    buffer1 = np.fromfile(f, bufferType, 1)[0]
    f.seek(buffer1 + buffer1.nbytes, 1)
    idSize = buffer1 / noParticles
    idType = intTypes[idSize]

    # read mass block
    massSize = 4
    if hasMass:
        f.seek(offsetSize, 1)
        buffer1 = np.fromfile(f, bufferType, 1)[0]
        f.seek(buffer1 + buffer1.nbytes, 1)
        massSize = buffer1 / noParticles
    massType = floatTypes[massSize]
    f.close()

    if VERBOSE:
        print(
            "Gadget data types: pos <-> ",
            posType,
            "  vel <-> ",
            velType,
            "  id <-> ",
            idType,
            # end=" ",
        )
        if hasMass:
            print("  mass <-> ", massType)
        else:
            print("")
    return (
        posType,
        posSize,
        velType,
        velSize,
        idType,
        idSize,
        massType,
        massSize,
    )


def readGadgetHeader(file, INDEX=0, VERBOSE=True):
    """   Reads only the Gadget header from the given Gadget file. It returns the results as the class 'GadgetHeader'.
    Takes as argument the name (or root anme) of the file from where to read the header.
    Can use VERBOSE = False to turn off the message."""
    header = GadgetHeader()
    tempName = file
    if not os.path.isfile(tempName):
        tempName = gadgetMultipleFiles(file, INDEX)
        if not os.path.isfile(tempName):
            throwError(
                "Cannot find the gadget snapshot file. There are no '%s' or '%s' files."
                % (file, tempName)
            )
    if VERBOSE:
        print("Reading the header of the Gadget file '%s' ... " % tempName)

    gadgetFileType, switchEndian, offsetSize = getGadgetFileType(tempName)

    f = open(tempName, "rb")
    f.seek(offsetSize, 1)
    header.fromfile(f, switchEndian=switchEndian)
    f.close()
    return header


def readGadgetData(
    file,
    HEADER=True,
    POS=True,
    VEL=True,
    ID=True,
    MASS=True,
    VERBOSE=True,
    NO_FILES=-1,
):
    """ Reads the header and data in a Gadget snapshot (single or multiple files).
    It returns the class 'GadgetParticles' which keeps track of the gadget header (class 'GadgetHeader' - if HEADER=True) and the data in numpy array/arrays (depending on what data to read).
    Can choose what options to choose via the boolean parameters (default ALL True): HEADER = True/False; POS; VEL; ID; MASS;
    Can use VERBOSE = False to turn off the messages."""
    if VERBOSE:
        print(
            "\nReading the data in the Gadget snapshot file/s '%s' ... "
            % (file)
        )
        print("This functions reads only particle type 1 data!")

    # Gadget snapshot file variables
    NO_DIM = 3  # the number of spatial dimensions
    bufferType = np.dtype("i4")  # the buffer type before each block of data
    bufferSize = np.array(0, bufferType).nbytes  # the buffer size

    # read the header to find in how many files the snapshot is
    tempName = file
    if not os.path.isfile(tempName):
        tempName = gadgetMultipleFiles(file, 0)
        if not os.path.isfile(tempName):
            throwError(
                "Cannot find the gadget snapshot file. There are no '%s' or '%s' files."
                % (file, tempName)
            )

    gadgetFileType, switchEndian, offsetSize = getGadgetFileType(tempName)
    firstFile = tempName

    header = readGadgetHeader(file, VERBOSE=False)
    if NO_FILES >= 1:
        header.num_files = NO_FILES
    if header.num_files > 1:
        for i in range(header.num_files):
            tempName = gadgetMultipleFiles(file, i)
            if not os.path.isfile(tempName):
                throwError(
                    "Cannot find the gadget snapshot file number %i with expected name '%s'."
                    % (i, tempName)
                )

    # variables where to store the data
    noTotParticles = gadgetTotalParticleCount(
        file, header.num_files, VERBOSE=VERBOSE
    )[1]
    hasMass = False
    if header.mass[1] == np.float64(0.0):
        hasMass = True

    (
        posType,
        posSize,
        velType,
        velSize,
        idType,
        idSize,
        massType,
        massSize,
    ) = gadgetDataType(
        firstFile,
        header.npart.sum(),
        hasMass,
        offsetSize,
        switchEndian,
        bufferType=bufferType,
        VERBOSE=VERBOSE,
    )

    positions, velocities, ids, masses = None, None, None, None
    if POS:
        positions = np.empty(NO_DIM * noTotParticles, posType)
    if VEL:
        velocities = np.empty(NO_DIM * noTotParticles, velType)
    if ID:
        ids = np.empty(noTotParticles, idType)
    if MASS and hasMass:
        masses = np.empty(noTotParticles, massType)

    # read the data from each file
    startPosition = 0
    for i in range(header.num_files):
        tempName = file
        if not os.path.isfile(tempName):
            tempName = gadgetMultipleFiles(file, i)
        if not os.path.isfile(tempName):
            throwError(
                "Cannot find the gadget snapshot file number %i with expected name '%s'."
                % (i, tempName)
            )
        if VERBOSE:
            print(
                "Reading the data in the Gadget file '%s' which is file %i of %i files ... "
                % (tempName, i + 1, header.num_files)
            )

        # read the header
        f = open(tempName, "rb")
        tempHeader = GadgetHeader()
        f.seek(offsetSize, 1)
        tempHeader.fromfile(
            f, BUFFER=True, bufferType=bufferType, switchEndian=switchEndian
        )
        dataSize = tempHeader.npart[1]
        skipBefore, skipAfter = tempHeader.npart[0], tempHeader.npart[2:6].sum()

        # read the positions
        f.seek(offsetSize, 1)
        if POS:
            __buffer1 = np.fromfile(f, bufferType, 1)[0]
            f.seek(skipBefore * posSize * 3, 1)
            positions = readArrayEntries(
                positions, f, NO_DIM * startPosition, NO_DIM * dataSize
            )
            f.seek(skipAfter * posSize * 3, 1)
            __buffer2 = np.fromfile(f, bufferType, 1)[0]
            if __buffer1 != __buffer2:
                throwError(
                    "While reading the position block in file '%s'. The buffers before (value=%i) and after (value=%i) the data block do not agree."
                    % (tempName, __buffer1, __buffer2)
                )
        else:
            f.seek(NO_DIM * posSize * dataSize + 2 * bufferSize, 1)

        # read the velocities
        f.seek(offsetSize, 1)
        if VEL:
            __buffer1 = np.fromfile(f, bufferType, 1)[0]
            f.seek(skipBefore * velSize * 3, 1)
            velocities = readArrayEntries(
                velocities, f, NO_DIM * startPosition, NO_DIM * dataSize
            )
            f.seek(skipAfter * velSize * 3, 1)
            __buffer2 = np.fromfile(f, bufferType, 1)[0]
            if __buffer1 != __buffer2:
                throwError(
                    "While reading the velocities block in file '%s'. The buffers before (value=%i) and after (value=%i) the data block do not agree."
                    % (tempName, __buffer1, __buffer2)
                )
        else:
            f.seek(NO_DIM * posSize * dataSize + 2 * bufferSize, 1)

        # read the identities
        f.seek(offsetSize, 1)
        if ID:
            __buffer1 = np.fromfile(f, bufferType, 1)[0]
            f.seek(skipBefore * idSize, 1)
            ids = readArrayEntries(ids, f, startPosition, dataSize)
            f.seek(skipAfter * idSize, 1)
            __buffer2 = np.fromfile(f, bufferType, 1)[0]
            if __buffer1 != __buffer2:
                throwError(
                    "While reading the identities block in file '%s'. The buffers before (value=%i) and after (value=%i) the data block do not agree."
                    % (tempName, __buffer1, __buffer2)
                )
        else:
            f.seek(idSize * dataSize + 2 * bufferSize, 1)

        # read the masses if any
        if MASS and tempHeader.mass[1] == np.float64(0.0):
            s = tempHeader.mass == np.float64(0.0)
            skipBefore = s[0] * skipBefore
            skipAfter = (s[2:6] * tempHeader.npart[2:6]).sum()

            f.seek(offsetSize, 1)
            f.seek(skipBefore * massSize, 1)
            __buffer1 = np.fromfile(f, bufferType, 1)[0]
            masses = readArrayEntries(masses, f, startPosition, dataSize)
            f.seek(skipAfter * massSize, 1)
            __buffer2 = np.fromfile(f, bufferType, 1)[0]
            if __buffer1 != __buffer2:
                throwError(
                    "While reading the masses block in file '%s'. The buffers before (value=%i) and after (value=%i) the data block do not agree."
                    % (tempName, __buffer1, __buffer2)
                )
        f.close()
        startPosition += dataSize

    # output the results
    gadgetParticles = GadgetParticles()
    if HEADER:
        gadgetParticles.AddHeader(header)
    if POS:
        gadgetParticles.AddPos(positions)
    if VEL:
        gadgetParticles.AddVel(velocities)
    if ID:
        gadgetParticles.AddIds(ids)
    if MASS:
        gadgetParticles.AddMass(masses)
    return gadgetParticles


def writeGadgetData(file, gadgetParticles, VERBOSE=True):
    """ Writes a single Gadget file.
    It takes two arguments: the name of the output file and the data to be written as the class 'GadgetParticles'. """

    bufferType = np.dtype("i4")  # the buffer type before each block of data

    # do some error checking
    if VERBOSE:
        print("Writing the gadget data to the file '%s' ... " % file)
    if not gadgetParticles.CheckDataCompletness():
        throwError(
            "Cannot continue writing the Gadget snapshot since the data failled the completness test."
        )
    (noParticles, sameSize) = gadgetParticles.SameNumberOfParticles()
    if not sameSize:
        throwError(
            "Cannot continue writing the Gadget snapshot since the not all the Gadget particle data arrays have the same (or expected) size."
        )
    if gadgetParticles.Header().npartTotal[1] != noParticles:
        throwError(
            "Cannot continue writing the Gadget snapshot since the length of the Gadget particle data does is not the same as the expected nuber of particles from the Gadget header. Length of Gadget particle data is %i versus expected size %i (from the header)."
            % (noParticles, gadgetParticles.Header().npartTotal[1])
        )
    gadgetParticles.Header().num_files = np.int32(1)

    # write the header to the file
    f = open(file, "wb")
    gadgetParticles.Header().tofile(f)

    # write the data to the file
    pos = gadgetParticles.Pos().reshape(-1)
    pos.shape = -1
    __buffer = np.zeros(1, dtype=bufferType)
    __buffer[0] = pos.size * pos[0].nbytes
    __buffer.tofile(f)
    pos.tofile(f)
    __buffer.tofile(f)

    pos = gadgetParticles.Vel().reshape(-1)
    __buffer[0] = pos.size * pos[0].nbytes
    __buffer.tofile(f)
    pos.tofile(f)
    __buffer.tofile(f)

    pos = gadgetParticles.Ids().reshape(-1)
    __buffer[0] = pos.size * pos[0].nbytes
    __buffer.tofile(f)
    pos.tofile(f)
    __buffer.tofile(f)

    if gadgetParticles.hasMass:
        pos = gadgetParticles.Mass().reshape(-1)
        __buffer[0] = pos.size * pos[0].nbytes
        __buffer.tofile(f)
        pos.tofile(f)
        __buffer.tofile(f)

    f.close()
    if VERBOSE:
        print("Done.")


def gadgetCombine(p1, p2):
    """Combines two GadgetParticle classes into a single one. """
    nTotal = p1.noParticles + p2.noParticles
    res = GadgetParticles()
    res.AddHeader(p1.header)
    res.AddPos(np.append(p1.pos, p2.pos))
    res.AddVel(np.append(p1.vel, p2.vel))
    res.AddIds(np.append(p1.ids, p2.ids))
    if p1.VariableMass():
        res.AddMass(np.append(p1.mass, p2.mass))
    res.header.npart[1] = nTotal
    res.header.npartTotal[1] = nTotal
    return res


def boxFullyContained(mainBox, smallBox):
    """Checks if the 'smallBox' is fully contained inside 'mainBox'."""
    for i in range(3):
        if not (
            smallBox[2 * i] >= mainBox[2 * i]
            and smallBox[2 * i] <= mainBox[2 * i + 1]
        ):
            return False
        if not (
            smallBox[2 * i + 1] >= mainBox[2 * i]
            and smallBox[2 * i + 1] <= mainBox[2 * i + 1]
        ):
            return False
    return True


def boxOverlap(box1, box2):
    """Checks if the two boxes overlap."""
    for i in range(3):
        if box1[2 * i] >= box2[2 * i + 1] or box1[2 * i + 1] <= box2[2 * i]:
            return False
    return True


def selectParticlesInBox(
    particles, box, periodicLength, periodic=True, VERBOSE=True
):
    """Selects only the particle positions in the box of interest. It uses periodic boundary conditions to translate the box to outside its regions if the region of interest is partially outside the periodic box."""
    if VERBOSE:
        print("\nFinding the particles in the subBox %s ..." % str(box))
    mainBox = 0.0, periodicLength, 0.0, periodicLength, 0.0, periodicLength
    pos = particles.pos

    # code for when the box is fully contained in the periodic one
    if boxFullyContained(mainBox, box) or not periodic:
        particleInside = (
            (pos[:, 0] >= box[0])
            * (pos[:, 0] <= box[1])
            * (pos[:, 1] >= box[2])
            * (pos[:, 1] <= box[3])
            * (pos[:, 2] >= box[4])
            * (pos[:, 2] <= box[5])
        )
        particles.Update(particleInside)
        if VERBOSE:
            print(
                "\tfound %i particles (%.2f%%) out of the total of %i"
                % (
                    particles.pos.shape[0],
                    100.0 * particles.pos.shape[0] / pos.shape[0],
                    pos.shape[0],
                )
            )
        return particles

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
                intersect[index] = boxOverlap(mainBox, tempBox)
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
    result = GadgetParticles()
    result.AddHeader(particles.header)
    result.AddPos(np.empty((nTotal, 3), np.float32))
    result.AddVel(np.empty((nTotal, 3), np.float32))
    result.AddIds(np.empty(nTotal, np.int32))
    if particles.VariableMass():
        result.AddMass(np.empty(nTotal, np.float32))

    # loop again and copy the required particles
    count = 0
    No = 0
    for i1 in range(-1, 2):
        for i2 in range(-1, 2):
            for i3 in range(-1, 2):
                index = (i1 + 1) * 9 + (i2 + 1) * 3 + (i3 + 1)
                if not intersect[index]:
                    continue

                result.pos[No : No + n[index], 0] = (
                    particles.pos[select[count], 0] - i1 * periodicLength
                )
                result.pos[No : No + n[index], 1] = (
                    particles.pos[select[count], 1] - i2 * periodicLength
                )
                result.pos[No : No + n[index], 2] = (
                    particles.pos[select[count], 2] - i3 * periodicLength
                )
                result.vel[No : No + n[index], :] = particles.vel[
                    select[count], :
                ]
                result.ids[No : No + n[index]] = particles.ids[select[count]]
                if particles.VariableMass():
                    result.mass[No : No + n[index]] = particles.mass[
                        select[count]
                    ]

                count += 1
                No += n[index]
    if VERBOSE:
        print(
            "\tfound %i particles (%.2f%%) out of the total of %i"
            % (nTotal, 100.0 * nTotal / pos.shape[0], pos.shape[0])
        )
    return result
