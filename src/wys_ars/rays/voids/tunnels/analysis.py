import numpy as np
import math
import scipy.stats as stats

# ~ from scipy.weave import inline, converters
from wys_ars.rays.voids.tunnels.miscellaneous import throwError, throwWarning


dtype2ctype = {
    np.dtype(np.float64): "double",
    np.dtype(np.float32): "float",
    np.dtype(np.int32): "int",
    np.dtype(np.uint64): "size_t",
    np.dtype(np.int16): "short",
}


def MinMax(data):
    """ Computes the min and maximum of a data set. It return two tuples: (minValue,maxValue) and (minIndex,maxIndex). """
    if data.size <= 0:
        throwError(
            "The numpy array supplied as argument to the 'MinMax' function must have at least one elemnt."
        )

    dataType = dtype2ctype[data.dtype]

    # C++ code to efficiently compute the min and max of the data entries
    MinMaxCode = """
    #line 1000 "MinMaxCode"
    %s min = data(0), max = data(0);
    size_t minIndex = 0, maxIndex = 0;
    for (int i=1; i<data.size(); ++i)
    {
        if (data(i)<min) {min=data(i); minIndex=i;}
        if (max<data(i)) {max=data(i); maxIndex=i;}
    }
    values(0) = min; values(1) = max;
    indices(0) = minIndex; indices(1) = maxIndex; """ % (
        dataType
    )

    # Call the C++ code computing the min and max values
    _a = np.array([0, 0])
    values = _a.astype(data.dtype)
    indices = _a.astype(np.uint64)
    finalShape = data.shape
    data.shape = -1
    inline(MinMaxCode, ["data", "values", "indices"], type_converters=converters.blitz)
    data.shape = finalShape

    # return the results
    return [(values[0], values[1]), (indices[0], indices[1])]


def binBoundaries(dataRange, no_bins, bin_type="linear"):
    """ Returns the bin boundaries for 'no_bins' equally spaced in linear/logarithmic interval in the given range."""
    if bin_type not in ["linear", "logarithm"]:
        throwError(
            "Unknown bin type in function 'binBoundaries'. You inserted 'bin_type'='%s', but the function recognizes only the following values '%s'."
            % (bin_type, str(["linear", "logarithm"]))
        )
    output = None
    if bin_type is "linear":
        output = np.linspace(dataRange[0], dataRange[1], num=no_bins + 1, endpoint=True)
    if bin_type is "logarithm":
        output = np.logspace(
            np.log10(dataRange[0]),
            np.log10(dataRange[1]),
            num=no_bins + 1,
            endpoint=True,
        )
    return output


def binValues(dataRange, no_bins, bin_type="linear"):
    """ Returns the bin values for 'no_bins' equally spaced in linear/logarithmic interval in the given range."""
    if bin_type not in ["linear", "logarithm"]:
        throwError(
            "Unknown bin type in function 'binValues'. You inserted 'bin_type'='%s', but the function recognizes only the following values '%s'."
            % (bin_type, str(["linear", "logarithm"]))
        )
    binBoundary = binBoundaries(dataRange, no_bins, bin_type=bin_type)
    output = np.zeros(no_bins, binBoundary.dtype)
    if bin_type is "linear":
        output[:] = (binBoundary[0:no_bins] + binBoundary[1 : no_bins + 1]) / 2.0
    if bin_type is "logarithm":
        output[:] = np.sqrt(binBoundary[0:no_bins] * binBoundary[1 : no_bins + 1])
    return output


def Histogram(
    data, no_bins=100, bins=None, Range=(None, None), bin_type="linear", weights=None
):
    """ This function computes the histogram of an array of data.
        no_bins = the number of bins if bins are not given in the 'bins' array
        bins = custom array giving the bin boundaries (one more element than the number of bins)
        range = (min,max)
        bin_type = 'linear' or 'logarithm' for linear/logarithmic bin types."""

    dataType = dtype2ctype[data.dtype]

    # if the program must generate the bin boundaries
    if bins is None:
        if no_bins <= 0:
            throwError(
                "The 'no_bins' parameter of function 'Histogram' must be at least 1."
            )
        if bin_type not in ["linear", "logarithm"]:
            throwError(
                "Unrecognized histogram '%s' bin type in function 'Histogram'. The parameter 'bin_type' can be 'linear' or 'logarithm' to have linearly/logarithmically spaced bins."
                % bin_type
            )
        bins = np.zeros(no_bins, data.dtype)
        histData = np.zeros(no_bins, np.uint64)
        summationRule = """histData(temp) += weights(i)"""
        if weights is None:
            summationRule = """histData(temp) += 1"""

        # get the min and max ranges for the histogram
        MinRange, MaxRange = Range
        if MinRange is None or MaxRange is None:
            _min, _max = MinMax(data)[0]
            if MinRange is None:
                MinRange = _min * 0.99
            if MaxRange is None:
                MaxRange = _max * 1.01

        # code for computing the linear bin histogram
        HistogramCode_linearBin = """
        #line 1000 "HistogramCode_linearBin"
        int noBins = int(no_bins);
        %s minR=MinRange, maxR=MaxRange, dx = (maxR-minR)/noBins;
        for (int i=0; i<noBins; ++i)
        {
            bins(i) = minR + (i+0.5)*dx;
            histData(i) = 0;
        }
        for (int i=0; i<data.size(); ++i)
        {
            int temp = int( std::floor((data(i)-minR)/dx) );
            if ( temp>=0 and temp<noBins ) %s;
        }""" % (
            dataType,
            summationRule,
        )
        HistogramCode_logarithmicBin = """
        #line 1000 "HistogramCode_logarithmicBin"
        int noBins = no_bins;
        %s minR=MinRange, maxR=MaxRange, dx = std::log10(maxR/minR)/noBins;
        for (int i=0; i<noBins; ++i)
        {
            bins(i) = minR * std::pow( 10, (i+0.5)*dx );
            histData(i) = 0;
        }
        for (int i=0; i<data.size(); ++i)
        {
            int temp = int( std::floor(std::log10(data(i)/minR)/dx) );
            if ( temp>=0 and temp<noBins ) %s;
        }""" % (
            dataType,
            summationRule,
        )

        # call the C++ code on the data
        finalShape = data.shape
        if data.size != 0:
            data.shape = -1
        if weights is not None and weights.size is not data.size:
            throwError(
                "The 'data' and 'weights' input arrays of the 'Histogram' function must have the same size. 'data.size'=%s and 'weights.size'=%s."
                % (str(data.size), str(weights.size))
            )
        if bin_type is "linear":
            inline(
                HistogramCode_linearBin,
                [
                    "data",
                    "MinRange",
                    "MaxRange",
                    "no_bins",
                    "bins",
                    "histData",
                    "weights",
                ],
                type_converters=converters.blitz,
                headers=["<cmath>"],
            )
        if bin_type is "logarithm":
            if MinRange * MaxRange <= 0.0:
                throwError(
                    "Cannot take logarithmic bin steps when the product of the minimum and maximum bin limits is 0 or negative. 'MinRange'=%s and 'MaxRange'=%s."
                    % (str(MinRange), str(MaxRange))
                )
            inline(
                HistogramCode_logarithmicBin,
                [
                    "data",
                    "MinRange",
                    "MaxRange",
                    "no_bins",
                    "bins",
                    "histData",
                    "weights",
                ],
                type_converters=converters.blitz,
                headers=["<cmath>"],
            )
        if data.size != 0:
            data.shape = finalShape

    else:
        histData, bins = np.histogram(data, bins=bins, range=Range, weights=weights)

    return (histData, bins)


def CumulativeSum(data, order="ascending", out_type=None):
    """ Computes the cumulative sum of the entries of an array. 
    order = "ascending"/"descending" to sum starting with the first/last array element
    out_type - output data type of the resulting array """

    if order not in ["ascending", "descending"]:
        throwError(
            "The cumulative ordering '%s' is not recognized in function 'CumulativeSum'. The function recognizes only the values 'ascending'/'descending' to compute the ascending/descending cumulative sum."
            % str(order)
        )

    result = None
    if order is "ascending":
        result = np.cumsum(data, dtype=out_type)
    if order is "descending":
        dataShape = data.shape
        data.shape = -1
        temp = data[::-1]
        data.shape = dataShape
        temp2 = np.cumsum(temp, dtype=out_type)
        result = temp2[::-1]

    return result


def PDF(data, pdfRange=(None, None), bin_type="linear"):
    """ Computes the probability distribution function (PDF) for a 1D data with linear/logarithmic bins.
    range = min and max values for the bins range (number of bins = data set size)
    bin_type = the type of the bin: linear / logarithm / variable*
    NOTE *: For 'variable' bin types the range must be the bin boundaries used to compute the data - needs to have size = data.size+1."""

    bin_type_list = ["linear", "logarithm", "variable"]
    if data.ndim != 1:
        throwError("The data array in function 'PDF' must be a 1D numpy array.")
    (bins,) = data.shape
    if (pdfRange[0] is None) or (pdfRange[1] is None):
        throwError(
            "You must specify the bin limits for the data in function 'PDF' using the 'pdfRange' argument."
        )

    # get the bin boundaries
    binsSpacing = np.zeros(bins, np.float64)
    if bin_type is "linear":
        binsSpacing[:] = (pdfRange[1] - pdfRange[0]) / bins
    elif bin_type is "logarithm":
        binsSpacing[:] = (math.log10(pdfRange[1]) - math.log10(pdfRange[0])) / bins
    elif bin_type is "variable":
        if pdfRange.size != data.size + 1:
            throwError(
                "In function 'PDF'. For bin_types='variable' the 'pdfRange' argument should give the bin boundaries used to compute the data. So the 'pdfRange' argument should have length 'data'.size+1. But length('pdfRange')=%i while length('data')=%i."
                % (pdfRange.size, data.size)
            )
        binsSpacing[:] = pdfRange[1:] - pdfRange[:-1]
    else:
        throwError(
            "In function 'PDF'. Unknown value for the 'bin_type' argument. Allowed values are %s."
            % bin_type_list
        )

    # compute the PDF
    dataSum = np.sum(data)
    result = data / (dataSum * binsSpacing)

    return result


def Bootstrap(data):
    """Returns a bootsrapped version of the input data."""
    length = data.size
    indices = np.random.randint(0, length, length)
    return data.reshape(-1)[indices]


def Median(data, N=100, minSize=5):
    """Computes the median of the data set and the errors associated to the media determination using bootstrap. It generates 'N' different bootsrap realizations."""
    funcName = "Median"
    if data.ndim != 1:
        throwError("The '%s' functions works only for a 1D data set." % funcName)
    median = np.median(data)
    if (
        data.size < minSize
    ):  # the data is not large enough to get an accurate error for it
        return (median, median)
    medianSet = np.zeros(N, np.float64)
    for i in range(N):
        medianSet[i] = np.median(Bootstrap(data))
    medianError = np.std(medianSet)
    return (median, medianError)


def Average(data, N=100, minSize=5):
    """Computes the average of the data set and the errors associated to the arithmetic average determination using bootstrap. It generates 'N' different bootsrap realizations."""
    funcName = "Average"
    if data.ndim != 1:
        throwError("The '%s' functions works only for a 1D data set." % funcName)
    mean = np.mean(data)
    if (
        data.size < minSize
    ):  # the data is not large enough to get an accurate error for it
        return (mean, mean)
    meanSet = np.zeros(N, np.float64)
    for i in range(N):
        meanSet[i] = np.mean(Bootstrap(data))
    meanError = np.std(meanSet)
    return (mean, meanError)


def Percentile(data, percentile=(25.0, 75.0), N=100):
    """Computes the value of the percentiles of a distribution. It uses bootstrap to compute the error associated to the percentile determination.
    It returns a list of tuples, with each tupple giving the percentile value and the bootstrap error associated in determining it."""
    funcName = "Percentile"
    if data.ndim != 1:
        throwError("The '%s' functions works only for a 1D data set." % funcName)
    result = []
    for p in percentile:
        score = stats.scoreatpercentile(data, p)
        scoreSet = np.zeros(N, np.float64)
        for i in range(N):
            scoreSet[i] = stats.scoreatpercentile(Bootstrap(data), p)
        scoreError = np.std(scoreSet)
        result.append((score, scoreError))
    return result


def DistributionPercentile(X, Y, percentile=(25.0, 75.0)):
    """Computes the value of X for which the cumulative sum of Y has the requested percentile values. """
    res = CumulativeSum(Y)
    res = res / float(res[-1])
    indices = np.arange(res.size)
    result = []
    for per in percentile:
        p = per / 100.0
        i1 = (indices[res < p])[-1]
        i2 = i1 + 1
        if i2 == res.size:
            result.append(X[i1])
            continue
        slope = (X[i2] - X[i1]) / (res[i2] - res[i1])
        constant = X[i2] - slope * res[i2]
        result.append(slope * p + constant)
    return result


def CorrelationMatrix(data, N=100, minColumns=5):
    """Computes the correlation of the columns of the data set and the errors associated to the correlation coefficient using bootstrap. It generates 'N' different bootsrap realizations."""
    funcName = "CorrelationMatrix"
    if data.ndim != 2:
        throwError("The '%s' functions works only for a 2D data set." % funcName)
    correlation = np.corrcoef(data, rowvar=0)
    if (
        data.shape[0] < minColumns
    ):  # the data is not large enough to get an accurate error for it
        return correlation, correlation
    correlationSet = np.zeros(
        (N, correlation.shape[0] * correlation.shape[1]), np.float64
    )
    indices = np.arange(data.shape[0])
    for i in range(N):
        select = Bootstrap(indices)
        correlationSet[i, :] = np.corrcoef(data[select, :], rowvar=0).flatten()
    correlationError = np.std(correlationSet.reshape(N, -1), axis=0).reshape(
        correlation.shape[0], correlation.shape[1]
    )
    return correlation, correlationError


# def partialDependence(data,noSplits):
# """Computes the correlation in the data=(y,x1,x2) """


def magnitude(v):
    return np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2])


def scalarProduct(v1, v2):
    res = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1] + v1[:, 2] * v2[:, 2]
    res[:] /= magnitude(v1) * magnitude(v2)
    return res


def randomSample_direction(noPoints, noBins):
    X = np.random.rand(noPoints, 3) * 2.0 - 1.0
    Y = np.random.rand(noPoints, 3) * 2.0 - 1.0
    temp = np.abs(scalarProduct(X, Y))
    counts, discard = np.histogram(temp, bins=noBins, range=(0.0, 1.0))
    return counts


def randomCorrelation_direction(noPoints, noBins, noRepeats):
    temp = np.empty((noRepeats, noBins), np.float64)
    for i in range(noRepeats):
        temp[i, :] = randomSample_direction(noPoints, noBins)
    average = np.mean(temp, axis=0)
    spread = np.std(temp, axis=0)
    return average, spread


def dataCorrelation_direction(data, noBins, noRepeats):
    noPoints = data.shape[0]
    hist, discard = np.histogram(data, bins=noBins, range=(0.0, 1.0))
    temp = randomCorrelation_direction(noPoints, noBins, noRepeats)
    return hist, temp[0], temp[1]


def PCA(data):
    """Performs principal components analysis (PCA) on the n-by-p data matrix 'data'. Rows of 'data' correspond to observations, columns to variables. 
    Returns :  
        coeff : is a p-by-p matrix, each column containing coefficients for one principal component.
        score : the principal component scores; that is, the representation of 'data' in the principal component space. Rows of SCORE  correspond to observations, columns to components.
        latent :  a vector containing the eigenvalues of the covariance matrix of A. """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (data - np.mean(data.T, axis=1)).T  # subtract the mean (along columns)
    [latent, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M).T  # projection of the data in the new space
    return coeff, score, latent


def LeastSquare_linearFit(X, Y, constant=True, returnErrors=False):
    """Computes a linear fit using the least square method. See: http://mathworld.wolfram.com/LeastSquaresFitting.html"""
    xAvg, yAvg = X.mean(), Y.mean()
    ssXX = ((X - xAvg) ** 2).sum()
    ssYY = ((Y - yAvg) ** 2).sum()
    ssXY = ((X - xAvg) * (Y - yAvg)).sum()
    slope = ssXY / ssXX  # slope of function
    constant = yAvg - slope * xAvg  # constant of the fit
    r2 = ssXY * ssXY / (ssXX * ssYY)  # square of the correlation coefficient
    if returnErrors:
        n = X.size
        s = ((ssYY - slope * ssXY) / (n - 2.0)) ** 0.5
        slope_error = s / ssXX ** 0.5
        constant_error = s * (1.0 / n + xAvg * xAvg / ssXX) ** 0.5
        return slope, constant, r2, slope_error, constant_error
    return slope, constant, r2


def LeastSquare_general(Xs, Y, weights=None):
    """Computes a linear fit using the least square method. The 'Xs' is a list of numpy arrays of same size as 'Y' which give each of the X-parameters of the fit. Each such parameter will have a regresion coefficient 'a' and its error that will be computed by this function. For example, to fit a 2nd degree polynomial a0+a1*x+a2*x^2 than 'Xs' should contain 3 arrays giving: 1, x_i and x_i^2 .
    Returns: parameters, fit_error, chi_square, noPoints
    For details on the computation see: 
        http://ned.ipac.caltech.edu/level5/Stetson/Stetson_contents.html
        http://fityk.nieto.pl/fit.html
        http://mathworld.wolfram.com/LeastSquaresFitting.html
    """
    if weights is None:
        weights = Y.copy()
        weights[:] = 1
    noPoints = Y.size  # number of data points used for the fit
    noParams = len(Xs)  # number of fit parameters

    # find the linear system that needs to be solved by the least square method
    M = np.zeros((noParams, noParams), np.float64)
    V = np.zeros(noParams, np.float64)
    for i in range(noParams):
        V[i] = (weights * Xs[i] * Y).sum()
        for j in range(i, noParams):
            M[i, j] = (weights * Xs[i] * Xs[j]).sum()
            M[j, i] = M[i, j]

    # find the inverse of the M-matrix and find the best fit coefficients 'a'
    Minv = np.linalg.inv(M)
    a = np.dot(Minv, V)  # these are the best fit parameters

    # now get the standard error for the best fit parameters
    temp = a[0] * Xs[0]
    for i in range(1, noParams):
        temp[:] += a[i] * Xs[i]
    chiSquare = (weights * (Y - temp) ** 2).sum() / (
        noPoints - noParams
    )  # fit residuals
    a_error = np.zeros(noParams, np.float64)
    for i in range(noParams):
        a_error[i] = (chiSquare * Minv[i, i]) ** 0.5
    return a, a_error, chiSquare, noPoints


def LeastSquare_nonlinearFit_general(
    X,
    Y,
    func,
    func_derv,
    guess,
    weights=None,
    maxRelativeError=1.0e-5,
    maxIteratitions=100,
):
    """Computes a non-linear fit using the least square method following: http://ned.ipac.caltech.edu/level5/Stetson/Stetson2_2_1.html
    It takes the following arguments:
        X - array of x-parameters
        Y - y-array
        func - the function f(X,parameters)
        func_derv - a function that returns a 2D array giving along the columns the derivatives of the function 'f' with respect to each fit parameter ( df /dp_i )
        guess - a first guess for the fit parameters
        weights - the weights associated to each point
        maxRelativeError - stop the iteration once the error in each parameter is below this threshold
        maxIterations - stop the iteration after this many steps
    Returns: parameters, fit_error, chi_square, noPoints, succes (True if successful)
    """
    functionName = "'analysis.LeastSquare_nonlinearFit_general'"
    if weights is None:
        weights = Y.copy()
        weights[:] = 1
    noPoints = Y.size  # number of data points used for the fit
    noParams = len(guess)  # number of fit parameters

    # iterate starting with the initial guess until finding the best fit parameters
    a = guess
    iteration, notConverged = 0, True
    while iteration < maxIteratitions and notConverged:
        tempX = func_derv(X, a)  # the derivatives for the current parameter values
        tempDiff = Y - func(X, a)  # the difference between function values and Y-values
        std = (weights * tempDiff ** 2).sum()  # the current sum of the squares
        step = np.linalg.lstsq(tempX, tempDiff)[0]
        while True:
            a2 = a + step
            tempStd = (
                weights * (Y - func(X, a2)) ** 2
            ).sum()  # the sum of the squares for the new parameter values
            if tempStd > std:
                step /= 2.0  # wrong estimate for the step since it increase the deviation from Y values; decrease step by factor of 2
            else:
                a += step
                break
        if (np.abs(step / a) < maxRelativeError).all():
            notConverged = False  # the iteration has converged
        iteration += 1
        print(iteration, a, step, std, tempStd)

    # compute the standard deviation for the best fit parameters
    derivatives = func_derv(X, a)
    M = np.zeros((noParams, noParams), np.float64)
    for i in range(noParams):
        for j in range(i, noParams):
            M[i, j] = (weights * derivatives[:, i] * derivatives[:, j]).sum()
            M[j, i] = M[i, j]
    Minv = np.linalg.inv(M)
    chiSquare = (weights * (Y - func(X, a)) ** 2).sum() / (
        noPoints - noParams
    )  # fit residuals
    a_error = np.zeros(noParams, np.float64)
    for i in range(noParams):
        a_error[i] = (chiSquare * Minv[i, i]) ** 0.5
    return a, a_error, chiSquare, noPoints, iteration < maxIteratitions


def LeastSquare_nonlinearFit_singleParameter(
    X, Y, func, guess, relMaxError=1.0e-3, noMaxSteps=100
):
    """Computes the best fit for a nonlinear function using a trivial parameter space search."""
    param, step = guess, guess
    relError, iteration = 2 * relMaxError, 0
    pTry, error, index = np.zeros(5, X.dtype), np.zeros(5, X.dtype), np.arange(5)
    while iteration < noMaxSteps and relError > relMaxError:
        pTry[:] = param - step, param - step / 2, param, param + step / 2, param + step
        for i in range(5):
            temp = Y - func(X, pTry[i])
            error[i] = (temp * temp).sum()
        min = error.min()
        if min == error[0]:
            param = pTry[0]
        elif min == error[1]:
            param, step = pTry[1], step / 4
        elif min == error[2]:
            step = step / 4
        elif min == error[3]:
            param, step = pTry[3], step / 4
        elif min == error[4]:
            param = pTry[4]
        iteration += 1
        relError = step / param

    success = True
    if iteration == noMaxSteps:
        success = False
    errorSquare = ((Y - func(X, param)) ** 2).sum()
    return param, errorSquare, success


def LeastSquare_nonlinearFit(
    X, Y, func, deriv, guess, relMaxError=1.0e-3, noMaxSteps=100
):
    """Solves the nonlinear least square problem using the initial guess and the derivatives of the fit function with respect to the fit parameters. See for details: http://mathworld.wolfram.com/NonlinearLeastSquaresFitting.html"""
    funcName = "LeastSquare_nonlinearFit"
    if len(guess) != len(deriv):
        throwError(
            "In function '%s'. Inconsistent number of initial guess for the parameters and partial derivatives of the fit function. There are %i initial guess parameters and %i partial function derivatives."
            % (funcName, len(guess), len(deriv))
        )
    iteration, relError = 0, 10.0 * relMaxError
    noParam, param = len(guess), guess
    A = np.empty((len(X), len(guess)), np.float64)
    while iteration < noMaxSteps and relError > relMaxError:
        beta = Y - func(X, param)
        for i in range(noParam):
            A[:, i] = deriv[i](X, param)
        AT = A.transpose()
        a, b = np.dot(AT, A), np.dot(AT, beta)
        dParam = np.linalg.solve(a, b)
        relError = np.abs(dParam / param).max()  # maximum relative error
        param += dParam
        iteration += 1
        print(iteration, param, dParam, relError, beta)

    error = ((Y - func(X, param)) ** 2).sum()
    success = True
    if iteration == noMaxSteps:
        success = False
    return param, error, success


def PointDistribution(
    data, noBins=(10, 10), xRange=None, yRange=None, bin_type="linear"
):
    """ Computes the point density for a 2D data with linear/logarithmic bins.
            noBins = number of bins along each axis
            xRange = range extension along x-axis: (xmin,xmax)
            yRange = range extension along y-axis: (ymin,ymax)
            zRange = range extension along z-axis: (zmin,zmax) - if present
            bin_type = the type of the bin: linear/logarithm"""

    if bin_type not in ["linear", "logarithm"]:
        throwError(
            "Unknown bin type in function 'PointDistribution'. You inserted 'bin_type'='%s', but the function recognizes only the following values '%s'."
            % (bin_type, str(["linear", "logarithm"]))
        )
    if data.ndim != 2 or data.shape[1] != 2:
        throwError(
            "The input data in function 'PointDistribution' must be a 2D numpy array with 2 columns. The input data is a %i-D numpy array with %i columns."
            % (data.ndim, data.shape)
        )
    if xRange is None:
        xRange = (data.min(axis=0)[0], data.max(axis=0)[0])
    if yRange is None:
        yRange = (data.min(axis=0)[1], data.max(axis=0)[1])
    dataType = dtype2ctype[data.dtype]

    # code for computing the point density
    PointDistributionCode_linearBin = """
    #line 1000 "PointDistributionCode_linearBin"
    int const noDims = 2;
    int grid[noDims];
    %s minValue[noDims], maxValue[noDims], dx[noDims], weight = 1.;
    for (int i=0; i<noDims; ++i)
    {
        grid[i] = int(noBins(i));
        minValue[i] = Range(2*i);
        maxValue[i] = Range(2*i+1);
        dx[i] = (maxValue[i]-minValue[i]) / grid[i];
    }
    for (int i1=0; i1<grid[0]; ++i1)        // compute the grid point coordinates
        for (int i2=0; i2<grid[1]; ++i2)
        {
            bins(i1,i2,0) = minValue[0] + (i1+0.5)*dx[0];
            bins(i1,i2,1) = minValue[1] + (i2+0.5)*dx[1];
            bins(i1,i2,2) = 0.;    //stores the density
        }
    for (int i=0; i<data.size(); ++i)    // compute the number of points in each grid cell
    {
        bool valid = true;
        int index[noDims];
        for (int j=0; j<noDims; ++j)
        {
            index[j] = int( std::floor((data(i,j)-minValue[j])/dx[j]) );
            if ( not( index[j]>=0 and index[j]<grid[j] ) ) valid = false;
        }
        if (valid) bins(index[0],index[1],2) += weight;
    }""" % (
        dataType
    )
    PointDistributionCode_logarithmicBin = """
    #line 1000 "PointDistributionCode_logarithmicBin"
    int const noDims = 2;
    int grid[noDims];
    %s minValue[noDims], maxValue[noDims], dx[noDims], weight = 1.;
    for (int i=0; i<noDims; ++i)
    {
        grid[i] = int(noBins(i));
        minValue[i] = Range(2*i);
        maxValue[i] = Range(2*i+1);
        dx[i] = std::log10(maxValue[i]/minValue[i]) / grid[i];
    }
    for (int i1=0; i1<grid[0]; ++i1)        // compute the grid point coordinates
        for (int i2=0; i2<grid[1]; ++i2)
        {
            bins(i1,i2,0) = minValue[0] * std::pow( 10, (i1+0.5)*dx[0] );
            bins(i1,i2,1) = minValue[1] * std::pow( 10, (i2+0.5)*dx[1] );
            bins(i1,i2,2) = 0.;    //stores the density
        }
    for (int i=0; i<data.size(); ++i)    // compute the number of points in each grid cell
    {
        bool valid = true;
        int index[noDims];
        for (int j=0; j<noDims; ++j)
        {
            index[j] = int( std::floor(std::log10(data(i,j)/minValue[j])/dx[j]) );
            if ( not( index[j]>=0 and index[j]<grid[j] ) ) valid = false;
        }
        if (valid) bins(index[0],index[1],2) += weight;
    }""" % (
        dataType
    )

    # Call the C++ code that computes the point distribution
    noBins = np.array(noBins, np.int32)
    Range = np.array([xRange[0], xRange[1], yRange[0], yRange[1]], data.dtype)
    bins = np.zeros((noBins[0], noBins[1], 3), data.dtype)
    if bin_type is "linear":
        inline(
            PointDistributionCode_linearBin,
            ["data", "Range", "noBins", "bins"],
            type_converters=converters.blitz,
            headers=["<cmath>"],
        )
    elif bin_type is "logarithm":
        if MinRange * MaxRange <= 0.0:
            throwError(
                "Cannot take logarithmic bin steps when the product of the minimum and maximum bin limits is 0 or negative. 'MinRange'=%s and 'MaxRange'=%s."
                % (str(MinRange), str(MaxRange))
            )
        inline(
            PointDistributionCode_logarithmicBin,
            ["data", "Range", "noBins", "bins"],
            type_converters=converters.blitz,
            headers=["<cmath>"],
        )

    return bins


def FindContourValue(data, enclosed_fraction, bin_type="linear"):
    """Computes the contour level value for a 2D density map such that 'enclosed_fraction' of the total mass lies within the contour.
            enclosed_fraction should be an iterable object - can contain more than 1 value.
            Use bin_type = ['linear'.'logarithm'] to specify if the density values have a linear or logarithmic distribution - this determines the bin type used to compute the histogram that determines the required contour level."""
    if bin_type not in ["linear", "logarithm"]:
        throwError(
            "Unknown bin type in function 'FindContourValue'. You inserted 'bin_type'='%s', but the function recognizes only the following values '%s'."
            % (bin_type, str(["linear", "logarithm"]))
        )

    # Compute the histogram of the data
    temp = data.reshape(-1)
    noBins = 1000
    histogram, histogramBins = Histogram(
        temp, no_bins=noBins, bin_type=bin_type, weights=None
    )
    histogram = histogram * histogramBins
    histogram[0] = 0.0
    # Compute the cumulative sum
    cumSum = CumulativeSum(histogram, order="descending", out_type=np.float64)
    cumSum /= cumSum[0]
    # Find the bin value where the cumulative sum is >= 'enclosed_fraction'
    result = np.zeros((len(enclosed_fraction), 3))
    indices = np.arange(noBins)
    for i in range(len(enclosed_fraction)):
        validIndices = cumSum < enclosed_fraction[i]
        index = indices[validIndices][0] - 1
        result[i] = enclosed_fraction[i], histogramBins[index], cumSum[index]

    return result
