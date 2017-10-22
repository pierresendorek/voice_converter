def triangle(self, x, xLeftZero, xTop, xRightZero):
    """
    Given three points on the abscissa (x, xLeftZero, xRightZero),
    constructs a _triangle-shape function that is non-zero inside [xLeftZero, xRightZero]
    This is used an interpolation function which returns a value given x inside [xLeftZero, xRightZero]
    :param x: float point where the function is evaluated
    :param xLeftZero: abscissa of the left point of the _triangle
    :param xTop: abscissa of the top point of the _triangle
    :param xRightZero: abscissa of the right point of the _triangle
    :return: float
    """
    if x < xLeftZero or x > xRightZero:
        return 0
    elif x < xTop:
        return (x - xLeftZero) / (xTop - xLeftZero)
    else:
        return (xRightZero - x) / (xRightZero - xTop)


