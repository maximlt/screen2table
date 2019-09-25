"""Support module for screen2table.

It contains a class to hold and process the points clicked by
the user (ScreenData) and another one to hold the parameters
entered by the user (UserParam). It could have contained a class
for the cross-section and another one for the culvert (maybe
in the future).

Many functions are there to process culvert data and
create a level-width table.
"""
from collections import namedtuple
import math
import itertools
import configparser
import pathlib
import numpy as np
import matplotlib.path
from win32 import win32clipboard


# Initiliaze some data from the configs.ini file.
configs = configparser.ConfigParser()
configs.read(pathlib.Path(__file__).parent / "configs.cfg")
CLOSE_PTS_THRESHOLD = configs.getfloat("COMPUTATION", "CLOSE_PTS_THRESHOLD")

class ScreenData:
    """
    This class is a container for the points clicked by the user. It provides
    a method to transform those points into a polygon useful to compute
    a level-width table. It also provides a method to process the data
    recorded from the screen to plot a cross-section only.
    """

    def __init__(self, x, z, kind):
        self.x = x
        self.z = z
        self.kind = kind
        self.nb_points = len(self)
        self.dict_tkwarn = self.set_tkwarn_txt()
        self.is_ok = False
        self.xz = None
        self.xzinterp = None
        self.xz_to_plot = None

    def __len__(self):
        """Return the number of points clicked by the user."""
        return len(self.x)

    def set_tkwarn_txt(self):
        """Set a dict useful for prompting error messages for the screen data.

        Returns
        -------
        dict
            Dict of dicts.
        """
        tkwarn_txt = namedtuple('tkwarn_txt', ['title', 'error_message'])
        tkwarn_duplicates = tkwarn_txt(
            'Error',
            'Duplicate points not allowed.')
        if self.kind == "culvert":
            tkwarn_selfintersec = tkwarn_txt(
                'Error',
                'Self-intersecting polygon is not a valid geometry.')
            tkwarn_toofewpoints = tkwarn_txt(
                'Error',
                'Trace again a geometry with at least 3 points.')
            dict_tkwarn = {
                'toofewpoints': {
                    'is_error': False,
                    'tktxt': tkwarn_toofewpoints
                },
                'duplicates': {
                    'is_error': False,
                    'tktxt': tkwarn_duplicates
                },
                'self_intersection': {
                    'is_error': False,
                    'tktxt': tkwarn_selfintersec
                },
            }
        if self.kind == "xs":
            tkwarn_toofewpoints = tkwarn_txt(
                'Error',
                'Trace again a geometry with at least 2 points.')

            dict_tkwarn = {
                'toofewpoints': {
                    'is_error': False,
                    'tktxt': tkwarn_toofewpoints
                },
                'duplicates': {
                    'is_error': False,
                    'tktxt': tkwarn_duplicates
                }
            }

        return dict_tkwarn

    @staticmethod
    def detect_two_lines_intersection(p0, p1, p2, p3):
        """Detect whether two lines are intersecting.

        Adapted from
        https://stackoverflow.com/a/19550879/10875966

        Parameters
        ----------
        p0, p1, p2, p3: point coordinates.

        Returns
        -------
        bool
            True if intersection detected, else False.
        """
        s10_x = p1[0] - p0[0]
        s10_y = p1[1] - p0[1]
        s32_x = p3[0] - p2[0]
        s32_y = p3[1] - p2[1]
        denom = s10_x * s32_y - s32_x * s10_y
        if denom == 0:
            return False  # collinear

        denom_is_positive = denom > 0
        s02_x = p0[0] - p2[0]
        s02_y = p0[1] - p2[1]
        s_numer = s10_x * s02_y - s10_y * s02_x
        if (s_numer < 0) == denom_is_positive:
            return False  # no collision

        t_numer = s32_x * s02_y - s32_y * s02_x
        if (t_numer < 0) == denom_is_positive:
            return False  # no collision
        if (
            (s_numer > denom) == denom_is_positive or
            (t_numer > denom) == denom_is_positive
        ):
            return False  # no collision

        # collision
        return True

    def is_self_intersecting_polyg(self):
        """Check whether the clicked points form a self-intersecting polygon.

        Returns
        -------
        bool
            True if self-intersecting.
        """
        # Close the shape.
        xs = self.x + [self.x[0]]
        zs = self.z + [self.z[0]]
        # Create a list of two lines to be tested against each other.
        # Only non adjacent lines are tested.
        pts = list(zip(xs, zs))
        lines = list(zip(pts, pts[1:]))
        two_lines = list(itertools.combinations(lines, 2))
        two_lines_not_adjacent = []
        for line in two_lines:
            if line[0][1] != line[1][0]:
                if line[0][0] != line[1][1]:
                    two_lines_not_adjacent.append(line)
        for line1, line2 in two_lines_not_adjacent:
            if self.detect_two_lines_intersection(*line1, *line2):
                return True
        return False

    def process_screen_culvert(self):
        """Process the culvert coordinates entered by the user.

        Returns
        -------
        tuple
            Number of points clicked by the user -> int,
            Coordinates to plot -> np.ndarray,
            Interpolated polygon -> np.ndarray.
        """
        # Case when less than 3 points were clicked.
        # That also includes when only the stop button was clicked.
        if self.nb_points <= 2:
            self.dict_tkwarn['toofewpoints']['is_error'] = True
            return False

        # The user clicked more than 2 points, now we can check
        # the validity of the shape: self-intersecting polygons
        # are not allowed.
        if self.nb_points > 3:
            if self.is_self_intersecting_polyg():
                self.dict_tkwarn['self_intersection']['is_error'] = True
                return False

        # The screen coordinates are converted into a numpy array of floats...
        self.xz = np.vstack((self.x, self.z)).T
        self.xz = self.xz.astype(np.float64)

        # No duplicate points are allowed.
        if arr_contain_duplicates(self.xz):
            self.dict_tkwarn['duplicates']['is_error'] = True
            return False

        # ... whose z axis is flipped because the y screen coordinate
        # recorded thanks to pynput has its origin on the top left corner.
        xz_temp = flip(self.xz, 1)

        # Slightly modify the z values if there are points with the same
        # z on each side.
        if not are_all_z_unique(xz_temp):
            xz_temp = modify_equal_z(xz_temp)

        # Append the 1st row to plot the culvert as a closed shape.
        self.xz_to_plot = np.vstack((xz_temp, xz_temp[0, :]))

        # Interpolation on the X axis of the array.
        self.xzinterp = add_z_points_to_polygon(xz_temp)

        self.is_ok = True

    def process_screen_xs(self):
        """Process the cross-section coordinates entered by the user.

        Returns
        -------
        tuple
            Number of points clicked by the user -> int,
            Coordinates to plot -> np.ndarray.
        """
        # Case when less than 2 points were clicked.
        # That also includes when only the stop button was clicked.
        if self.nb_points <= 1:
            self.dict_tkwarn['toofewpoints']['is_error'] = True
            return False

        # The user clicked more than 2 points, now the screen coordinates
        # are converted into a numpy array of floats...
        self.xz = np.vstack((self.x, self.z)).T
        self.xz = self.xz.astype(np.float64)

        # No duplicate points are allowed.
        if arr_contain_duplicates(self.xz):
            self.dict_tkwarn['duplicates']['is_error'] = True
            return False

        # ... whose z axis is flipped because the y screen coordinate
        # recorded thanks to pynput has its origin on the top left corner.
        self.xz = flip(self.xz, 1)
        # The cross-section case is simple, the plotted output is
        # the same as the output copied to clipboard
        self.xz_to_plot = self.xz.copy()

        self.is_ok = True


def arr_contain_duplicates(array):
    """Check if there are duplicates rows in an array.

    Parameters
    ----------
    array : np.ndarray

    Returns
    -------
    bool
        True if duplicates are found.
    """
    if len(np.unique(array, axis=0)) != len(array):
        return True
    else:
        return False


def flip(array, col_num):
    """Flip one column of an array.

    This is a symetry around the axis and a translation.

    >>> flip(np.array([[0, 4], [1, 3],  [2, 2],]), 0)
    array([[2, 4], [1, 3], [0, 2]])
    >>> flip(np.array([[0, 4], [1, 3],  [2, 2],]), 1)
    array([[0, 0], [1, 1], [2, 2]])

    Parameters
    ----------
    array : np.ndarray
    col_num : int
        Index of the column to use to flip the array.

    Returns
    -------
    np.ndarray
        Flipped array.
    """
    array = array.copy()
    max_col = np.max(array[:, col_num])
    # Symetry (-) and translation (+)
    array[:, col_num] = - array[:, col_num] + max_col
    return array


# =============================================================================
# Supporting functions for polygon transformations
def are_all_z_unique(xz_in):
    """Check whether all the z values in a polygon are unique.

    Parameters
    ----------
    xz_in : np.ndarray
        X:1st column, Z:2nd column.

    Returns
    -------
    bool
        True if the z values are unique, else False.
    """
    # The function is designed to take polygons whose last point is not
    # equal to the first point.
    if np.all(xz_in[-1, :] == xz_in[0, :]):
        xz_in = xz_in[:-1, :]

    # Count the number of points in the original array
    # and the number of unique z values.
    nb_points = xz_in.shape[0]
    nb_unique_z = np.unique(xz_in[:, 1]).shape[0]

    # If those numbers are different, there aren't only unique z values in
    # the polygon.
    return bool(nb_points == nb_unique_z)


def modify_equal_z(xz_in):
    """
    Modify the z values of a polygon which has at least two points with
    the same z values. A small random number is added to each
    non-unique z value. The maximum and minimum of the original polygon is
    preserved.

    Parameters
    ----------
    xz_in : np.ndarray
        X:1st column, Z:2nd column

    Returns
    -------
    np.ndarray
        Modified array.
    """
    # The function is designed to process polygons whose last point is not
    # equal to the first point.
    if np.all(xz_in[-1, :] == xz_in[0, :]):
        xz_in = xz_in[:-1, :]

    # Get the list of unique z values and their count (>=2 if non unique).
    z_values, unique_counts = np.unique(xz_in[:, 1], return_counts=True)
    # Array of the non-unique z values.
    not_unique = z_values[unique_counts >= 2]
    # From wich a mask is created based on the original array, True where
    # it finds a non-unique z.
    mask_not_unique = np.isin(xz_in[:, 1], not_unique)
    # The indexes of those non unique values are retrieved.
    idx = np.where(mask_not_unique)[0]
    # The number of locations that have to be changed is calculated.
    nb_not_unique = idx.shape[0]
    # A small random value is added to each non unique value.
    xz_out = xz_in.copy()
    rand_arr = np.random.uniform(low=-0.999, high=0.999, size=nb_not_unique)
    xz_out[idx, 1] = xz_in[idx, 1] + rand_arr * 1e-3

    # The original minimum z is preserved by adding positive values only.
    idx_min = np.where(xz_in[:, 1] == xz_in[:, 1].min())[0]
    nb_min = idx_min.shape[0]
    if nb_min > 1:
        rand_min = np.random.uniform(0.001, 0.999, nb_min)
        xz_out[idx_min, 1] = xz_in[idx_min, 1] + rand_min * 1e-3
    # The original maximum z is preserved by adding negative values only.
    idx_max = np.where(xz_in[:, 1] == xz_in[:, 1].max())[0]
    nb_max = idx_max.shape[0]
    if nb_max > 1:
        rand_max = np.random.uniform(-0.999, -0.001, size=nb_max)
        xz_out[idx_max, 1] = xz_in[idx_max, 1] + rand_max * 1e-3

    return xz_out


def add_z_points_to_polygon(xz_in):
    """Add points to a polygon so that it has at least two points on each z
    level, except at the minimum and the maximum.

    Parameters
    ----------
    xz_in : np.ndarray
        X:1st column, Z:2nd column

    Returns
    -------
    np.ndarray
        Polygon with interpolated vertices.
    """
    # The function is designed to take polygons whose last point is
    # equal to the first point.
    if np.any(xz_in[-1, :] != xz_in[0, :]):
        xz_in = np.vstack((xz_in, xz_in[0, :]))

    # Instantiate the output polygon.
    xz_out = np.ndarray((0, 2))
    # Go through all the segments of the polygon.
    for i, (xi, zi) in enumerate(xz_in[:-1, :]):
        # Retrieve the coordinates of the segment.
        xj = xz_in[i + 1, 0]
        zj = xz_in[i + 1, 1]

        # Add the first point of the segment to the output polygon.
        xz_out = np.vstack((xz_out, xz_in[i, :]))

        # Check if there are points within the polygon whose z is located
        # between the zi and zj of the current segment.
        cond_sup = xz_in[:, 1] < max(zi, zj)
        cond_inf = xz_in[:, 1] > min(zi, zj)
        mask_z_to_add = np.logical_and(cond_sup, cond_inf)
        # Count the number of points found.
        nb_val_to_itp = sum(mask_z_to_add)
        # If at least one point was found.
        if nb_val_to_itp >= 1:
            # Instantiate an array of the new interpolated points.
            xz_itp = np.zeros((nb_val_to_itp, 2))
            # Add their Z values which are known.
            # Case when the segment is going upwards.
            # The Z values must be added in an ascending order.
            if zi < zj:
                xz_itp[:, 1] = np.sort(xz_in[:, 1][mask_z_to_add])
            # Case when the segment is going downwards.
            # The Z values must be added in an decreasing order.
            elif zi > zj:
                xz_itp[:, 1] = np.sort(xz_in[:, 1][mask_z_to_add])[::-1]
            # Given the condition to find the points to add, this case
            # (zi == zj) doest not exist.
            else:
                pass
            # Go through each new point to add.
            for k in range(nb_val_to_itp):
                # Get the Z value (for readability).
                z_itp = xz_itp[k, 1]
                # Interpolate the x value of the new point.
                xz_itp[k, 0] = linear_interp_pt(z_itp, zi, zj, xi, xj)
            # Add the interpolated points to the output polygon.
            xz_out = np.vstack((xz_out, xz_itp))

    # Add the last point (it's the same as the first point).
    xz_out = np.vstack((xz_out, xz_in[-1, :]))

    # The algorithm above can create duplicate points when
    # there are points on the same z level.
    # Check if there are duplicate created points
    if np.unique(xz_out, axis=0).shape[0] != xz_out.shape[0]:
        # And delete only the unnecessary points.
        _, idx = np.unique(xz_out, return_index=True, axis=0)
        xz_out = xz_out[np.sort(idx), :]
        # The last point is added again as it was just deleted in the process.
        xz_out = np.vstack((xz_out, xz_in[-1, :]))

    return xz_out


def linear_interp_pt(x_target, x0, x1, y0, y1):
    """Convenience function to compute an interpolated linear value between
    two points.

    Parameters
    ----------
    x_target : float
    x0 : float
    x1 : float
    y0 : float
    y1 : float

    Returns
    -------
    float
        Interpolated value.
    """
    coeff = (y1 - y0) / (x1 - x0)
    dist = x_target - x0
    y_target = y0 + coeff * dist
    return y_target


# =============================================================================
class UserParam:
    """
    This class contains the user's parameters to scale the culvert to its real
    dimensions. The methods allow for the calculation of the Level-Width table
    and the area.
    """
    def __init__(self, user_dict):
        self.user_dict = user_dict
        self.convert_dict = None

    def validate(self):
        """ Check the validity of the input parameters against a set of rule.

        If the rules are OK, it converts a the input user params to floats.

        Returns
        -------
        dict
            Info about whether the rules are followed or not.
        """
        is_float = True
        is_x_ok = True
        is_z_ok = True
        is_angle_ok = True
        for value in self.user_dict.values():
            try:
                float(value)
            except ValueError:
                is_float = False

        if is_float:
            # Convert the values to floats.
            self.convert_dict = {
                k: float(v)
                for k, v in self.user_dict.items()
            }

            # Check more rules.
            # maxx = minx and maxz = minz are possible for cross-sections
            # in the case of a horizontal/vertical straight line.
            if self.convert_dict['maxx'] < self.convert_dict['minx']:
                is_x_ok = False
            if self.convert_dict['maxz'] < self.convert_dict['minz']:
                is_z_ok = False
            if not (-180 <= self.convert_dict['angle'] <= 180):
                is_angle_ok = False
            if self.convert_dict['angle'] in [-90, 90]:
                is_angle_ok = False

        rule = namedtuple('rule', ['is_ok', 'err_message'])
        # Result with the error message.
        rules_dict = {
            'float': rule(is_float, 'Values must be floating points.'),
            'x': rule(is_x_ok, 'Values must follow: Max X > Min X.'),
            'z': rule(is_z_ok, 'Values must follow: Max Z > Min Z.'),
            'angle': rule(
                is_angle_ok, (
                    'Values must follow -180 <= Angle <= 180'
                    ' and Angle != [-90, 90]'
                    )
            ),
        }

        return rules_dict


# =============================================================================
def polygon_to_levelwidth_table(xz_interp, user_param):
    """Transform an interpolated polygon into a level-Width table, also
    return an array useful for plotting the table.

    Parameters
    ----------
    xz_interp : np.ndarray
        Polygon with interpolated Z vertices.
    user_param : dict
        User parameters.

    Returns
    -------
    tuple
        Level width table (np.ndarray),
        Plottable level-width table.
    """
    # Scale the xz data from pixel to real dimensions
    xz_interp_realdim = scale_to_realdim(xz_interp, user_param)

    # Transform the interpolated xz polygon into a height-width table
    hw_table = polygon_to_heightwidth_table(xz_interp_realdim)

    # Turn it into a level-width table
    zw_table = hw_table
    zw_table[:, 0] = zw_table[:, 0] + np.min(xz_interp_realdim[:, 1])

    # Remove points that are too close to each other in the
    # level width table (Mike doesn't like that apparently).
    zw_table = remove_close_points(zw_table)

    # Create a zw table suitable for plotting it.
    zw_plot = zw_to_plot(zw_table)

    return (zw_table, zw_plot)


def polygon_to_heightwidth_table(xz_in):
    """Compute a height-width table from a densified polygon.

    Compute a height-width table from a polygon of which the z levels
    have been densified/interpolated so that, except for the max
    and the min z points, each point has at least another point
    at the same level.

    The polygon can be of any shape (concave, convex) and orientation.

    Parameters
    ----------
    xz_in : numpy.ndarray
        Column 0: X values, Column 1: Z values.

    Returns
    -------
    numpy.ndarray
        Column 0: Height values, Column 1: Width values.
    """
    # Make sure the polygon is closed in the first place to define the path.
    if np.any(xz_in[-1, :] != xz_in[0, :]):
        xz_in = np.vstack((xz_in, xz_in[0, :]))

    # A Matplotlib path is created from the vertices of the polygon.
    polygon_path = matplotlib.path.Path(xz_in, closed=True)

    # Then open it because the function needs it to be this way.
    xz_in = xz_in[:-1, :]

    # Get the number of unique z levels.
    nb_z_unique = np.unique(xz_in[:, 1]).shape[0]
    # To create the output table.
    hw_out = np.zeros((nb_z_unique, 2), dtype=np.float64)
    # The minimum z is used to compute each height.
    minz = xz_in[:, 1].min()
    # A list of the z values seen in the loop below is created and will be
    # appended each time a new z level has been processed.
    list_z = []
    count = 0  # Count the number of z levels processed.
    for _, zi in xz_in:
        if zi not in list_z:
            # Calculate the height and add it to the table.
            hw_out[count, 0] = zi - minz
            # Get the number of points at the same z value.
            same_z = np.where(xz_in[:, 1] == zi)[0]
            nb_same_z = same_z.shape[0]
            # If there is only one point, it is either the minimum or the
            # maximum.
            if nb_same_z == 1:
                # Then the width is 0.
                hw_out[count, 1] = 0
            # If there are only two points, it's straightforward to compute
            # the distance between these points.
            if nb_same_z == 2:
                hw_out[count, 1] = np.abs(np.diff(xz_in[same_z, 0]))
            # Now it is more complicated.
            if nb_same_z >= 3:
                # Get the coordinates of the points.
                points = xz_in[same_z, :]
                # Order the array to get the points in a x-ascending order.
                points = points[points[:, 0].argsort()]
                # Calculate the distance between the x-ordered points.
                widths = np.diff(points[:, 0])
                # Compute the location of the n-1 centers
                # obtained from the points.
                centers = 0.5 * (points[:-1, :] + points[1:, :])
                w_total = 0
                # Only those centers which are located within the polygon
                # should be considered. Their associated widths are
                # cumulated.
                for center, width in zip(centers, widths):
                    if polygon_path.contains_point(center):
                        w_total += width
                hw_out[count, 1] = w_total

            # Append the list of the z values already processed and increment
            # the count.
            list_z.append(zi)
            # Increment the count.
            count += 1

    # Sort the table by the z values before returning it.
    hw_out = hw_out[hw_out[:, 0].argsort()]
    return hw_out


def scale_to_realdim(arr_in, param):
    """Scale a polygon or a cross-section.

    Parameters
    ----------
    arr_in : numpy.ndarray
        Column 0: X values, Column 1: Z values.
    param : dict
        Scaling and optional parameters, like
        'minx', 'maxx', 'minz', 'maxz', 'angle'.

    Returns
    -------
    numpy.ndarray
        Scaled data: Column 0: X values, Column 1: Z values.
    """
    # Normalize each dimension to [0, 1].
    # Move first the origin of the data to 0 (translation).
    arr_out = arr_in - np.min(arr_in, axis=0)
    # And finalize the normalization by dividing by the max.
    arr_out = arr_out / np.max(arr_out, axis=0)

    # Compute the scaling factors for both dimensions.
    xscale = (
        (param['maxx'] - param['minx'])
        * math.cos(math.radians(param['angle']))
    )
    zscale = param['maxz'] - param['minz']

    # Rescale to the required dimensions.
    arr_out[:, 0] = arr_out[:, 0] * xscale + param['minx']
    arr_out[:, 1] = arr_out[:, 1] * zscale + param['minz']

    return arr_out


def zw_to_plot(zw_real):
    """Transform a zw table in a table that can be plotted, as in Mike Hydro.

    Parameters
    ----------
    zw_real : np.ndarray
        Column 0: Z values, Column 1: W values.

    Returns
    -------
    np.ndarray
        Transformed z-w table.
    """
    zw_plot = np.append(zw_real[:-1, :],
                        zw_real[::-1, :],
                        axis=0)
    idxmiddle = len(zw_plot) // 2
    zw_plot[1:idxmiddle, 1] = - zw_plot[1:idxmiddle, 1] / 2
    zw_plot[idxmiddle+1:-1, 1] = + zw_plot[idxmiddle+1:-1, 1] / 2
    return zw_plot


def calc_area(zw_real):
    """ Calculate the area of a culvert.

    Formula:
    Area = Sum(0.5 * (w[i+1] + w[i]) * (z[i+1] - z[i]))

    Parameters
    ----------
    zw_real : np.ndarray
        First column: X, Second column: Z.

    Returns
    -------
    float
        Area of the culvert.
    """
    area = np.sum(
        np.diff(zw_real[:, 0]) * np.add(zw_real[1:, 1], zw_real[:-1, 1]) / 2
    )
    return area


def calc_length(xz):
    """Calculate the length of a polyline.

    Parameters
    ----------
    xz : np.ndarray
        First column: X, Second column: Z.

    Returns
    -------
    np.float64
        Length of the polyline.
    """
    return np.sum(np.sqrt(np.sum(np.power(np.diff(xz, axis=0), 2), axis=1)))


def remove_close_points(zw_table, threshold=CLOSE_PTS_THRESHOLD):
    """Remove close points in a z(w) table.

    Parameters
    ----------
    zw_table : np.ndarray
        First column: z ; Second column: w.

    Returns
    -------
    np.ndarray
        First column: z ; Second column: w.
    """
    cleaned_arr = zw_table.copy()
    while True:
        diff = np.diff(cleaned_arr, axis=0)
        indexes = []
        for i, (zdiff, wdiff) in enumerate(diff):
            if zdiff < threshold and wdiff < threshold:
                indexes.append(i+1)
        if indexes:
            cleaned_arr = np.delete(cleaned_arr, indexes, axis=0)
        else:
            break
    return cleaned_arr


def to_clipboard_for_excel(array, decimals=None):
    r"""Copy an array into a string format acceptable by Excel.

    It is possible to round the array.
    Columns separated by \t, rows separated by \r\n.

    From:
    https://stackoverflow.com/questions/22488566/how-to-paste-a-numpy-array-to-excel

    Parameters
    ----------
    array : np.ndarray
        Array (to be rounded and) copied to the clipboard.
    decimals : int, optional
        Number of decimals of the rounded output, by default None.
    """
    # Create a list of strings representing the array data.
    # The data is numerically rounded and limited accordingly in the output.
    line_strings = []
    for row in array:
        temp_row = np.round(row, decimals) if decimals is not None else row
        formatted_line = [
            f"{elmt:.{decimals}f}" if decimals is not None else str(elmt)
            for elmt in temp_row
        ]
        line_strings.append("\t".join(formatted_line).replace("\n", ""))
    # Merge the strings in one string formatted for Excel/Mike.
    array_string = "\r\n".join(line_strings)

    # Put string into clipboard (open, clear, set, close)
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardText(array_string)
    win32clipboard.CloseClipboard()
