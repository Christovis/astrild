#!/usr/bin/env python
import sys
import numpy as np
from astrild.rays.voids.tunnels import halo
from astrild.rays.voids.tunnels.gadget import GadgetHeader, readGadgetHeader
from astrild.rays.voids.tunnels.density import DensityHeader, readDensityHeader
from astrild.rays.voids.tunnels.MMF import (
    MMFHeader,
    readMMFHeader,
    getHeaderType,
)

file = sys.argv[1]
headerType = getHeaderType(file)
header = None
if headerType is "Gadget":
    header = readGadgetHeader(file)
elif headerType is "Density":
    header = readDensityHeader(file)
elif headerType is "MMF":
    header = readMMFHeader(file)
elif headerType is "Halo":
    header = halo.readHaloHeader(file)
else:
    print("Unknown file type. Cannot read the binary file header.")
    sys.exit(1)

header.PrintValues()
