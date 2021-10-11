from datetime import datetime, timedelta
import time
import itertools

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import iris
from iris.coords import DimCoord, AuxCoord
from iris.cube import Cube
from iris.aux_factory import HybridHeightFactory
from iris.analysis._interpolation import extend_circular_coord_and_data
import iris.analysis.trajectory
import iris.coord_systems
import iris.fileformats.pp
import cf_units
import cartopy.crs as ccrs
import iris.tests.stock


def create_cube():
    """
    Create a cube to test the interpolation algorithm
    """
    # create some random data
    n_t, n_z, n_y, n_x = (12, 10, 1000, 1200)
    arr = np.random.random((n_t, n_z, n_y, n_x))

    # create some dummy coordinates
    dt = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(n_t)]
    time_unit = cf_units.Unit("hours since 1970-01-01")
    time_coord = DimCoord(time_unit.date2num(dt), standard_name="time",
                          units=time_unit)

    # add dim coords
    model_level_coord = DimCoord(np.arange(n_z),
                                 standard_name="model_level_number",
                                 units=1)
    cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    lat_coord = DimCoord(np.linspace(-90, 90, n_y), standard_name="latitude",
                         units="degrees", coord_system=cs)
    lon_coord = DimCoord(np.linspace(-180, 180, n_x, endpoint=False), standard_name="longitude",
                         units="degrees", coord_system=cs, circular=True)

    # attributes
    attributes = {"attribute1": 0,
                  "attribute2": "hello"}

    cube = Cube(arr, standard_name=None,  
                long_name="cube_of_random_data",
                var_name=None,
                dim_coords_and_dims=[(time_coord, 0),
                                     (model_level_coord, 1),
                                     (lat_coord, 2),
                                     (lon_coord, 3)],
                attributes=attributes)

    # add some aux coords spanning various dimensions
    forecast_period = AuxCoord(np.arange(n_t), standard_name="forecast_period",
                               units=1)
    level_height = AuxCoord(np.linspace(0, 100, n_z), 
                            long_name="level_height", units="m")
    sigma = AuxCoord(np.linspace(0.9, 0.99, n_z), long_name="sigma", units=1)
    orography = AuxCoord(100 * np.random.random((n_y, n_x)),
                         standard_name="surface_altitude", units="m")
    frequency = AuxCoord([100], standard_name="sound_frequency", units="Hz")

    cube.add_aux_coord(forecast_period, 0)
    cube.add_aux_coord(level_height, 1)
    cube.add_aux_coord(sigma, 1)
    cube.add_aux_coord(orography, (2, 3))
    cube.add_aux_coord(frequency, None)

    # add an aux factory for altitude
    factory = HybridHeightFactory(cube.coord("level_height"),
                                  cube.coord("sigma"),
                                  cube.coord("surface_altitude"))
    cube.add_aux_factory(factory)

    return cube


def _product(*args, related_index=None):
    """
    Return the product of the given arguments
    """
    out = []
    if related_index == None:
        index = [i for i in range(len(args))]
        for combi in itertools.product(*args):
            candidate = np.concatenate([np.atleast_1d(_) for _ in combi]) 
            out.append(candidate[np.argsort(index)])
    else:
        index = [i for i in range(len(args)) if i not in related_index]
        index.extend(related_index)
        unrelated = [args[i] for i in range(len(args)) if i not in related_index]
        related = [args[i] for i in range(len(args)) if i in related_index]
        if len(set([len(el) for el in related])) > 1:
            raise ValueError("the related items must have the same length")
        unrelated.append(list(zip(*related)))
        for combi in itertools.product(*unrelated):
            candidate = np.concatenate([np.atleast_1d(_) for _ in combi]) 
            out.append(candidate[np.argsort(index)])
    return out


def _interpolate(cube, sample_points, **kwargs):
    """
    Extract a sub-cube at the given n-dimensional points.

    Args:

    * cube
        The source Cube.

    * sample_points
        A sequence of coordinate (name) - values pairs.

    For example::

        sample_points = [('latitude', [45, 45, 45]),
        ('longitude', [-60, -50, -40])]
        interpolated_cube = interpolate(cube, sample_points)

    """
    # Convert any coordinate names to coords
    points = []
    points_array = {}
    for coord, values in sample_points:
        if isinstance(coord, str):
            coord = cube.coord(coord)
        points.append((coord, np.atleast_1d(values)))
        points_array[cube.coord_dims(coord)[0]] = np.atleast_1d(values)
    sample_points = points

    # Do all value sequences have the same number of values?
    coord, values = sample_points[0]
    trajectory_size = len(values)
    for coord, values in sample_points[1:]:
        if len(values) != trajectory_size:
            raise ValueError("Lengths of coordinate values are inconsistent.")

    # Are the given coords all 1-dimensional?
    for coord, values in sample_points:
        if coord.ndim > 1:
                msg = (
                    "Cannot currently perform linear interpolation for "
                    "multi-dimensional coordinates."
                )
                raise iris.exceptions.CoordinateMultiDimError(msg)

    cube_data = cube.data

    coord_names = [el[0].name() for el in sample_points]
    coord_dims = [(name, cube.coord_dims(name)) for name in coord_names]

    # determine where the dim coords will be shifted to by the interpolation
    count = 0
    shifted_dim_coords = []
    for coord in cube.dim_coords:
        if coord.name() not in coord_names:
            shifted_dim_coords.append((coord, (cube.coord_dims(coord), count)))
            count += 1

    # determine any auxiliary coordinates/factories that span dimensions to be
    # interpolated over
    dims = []
    for coord in cube.dim_coords:
        circular = getattr(coord, "circular", False)
        if circular:
            crd_dim = cube.coord_dims(coord)[0]
            points, cube_data = extend_circular_coord_and_data(coord,
                                                               cube_data,
                                                               crd_dim)
            dims.append(points)
        else:
            dims.append(coord.points)
    interpolation_dims = []
    for name, dim in coord_dims:
        interpolation_dims.extend(dim)
    interpolation_dims = set(interpolation_dims)

    factories_to_remove = []
    factories_to_keep = []
    for fact in cube.aux_factories:
        fact_dims = fact.derived_dims(cube.coord_dims)
        if len(set(fact_dims).intersection(interpolation_dims)) > 0:
            factories_to_remove.append((fact, fact_dims))

    coords_to_remove = []
    coords_to_keep = []
    for coord in cube.coords():
        coord_dims = cube.coord_dims(coord)
        if len(set(coord_dims).intersection(interpolation_dims)) > 0:
            if coord not in cube.derived_coords:
                coords_to_remove.append((coord, coord_dims))
        else:
            if coord not in cube.dim_coords:
                coords_to_keep.append((coord, coord_dims))

    # interpolate the data
    f = RegularGridInterpolator(dims, cube_data, method="linear", **kwargs)
    combinations = []
    options = len(cube.shape) * [None]
    for i, dim in enumerate(dims):
        if i not in interpolation_dims:
            options[i] = dim
    for key, val in points_array.items():
        options[key] = val

    combinations= _product(*options, related_index=interpolation_dims)

    array = f(combinations)
    out_shape = [s for i, s in enumerate(cube.shape) if i not in interpolation_dims]
    out_shape.append(trajectory_size)
    array = array.reshape(out_shape)

    out_cube = iris.cube.Cube(array, standard_name=cube.standard_name,
                              long_name=cube.long_name,
                              var_name=cube.var_name,
                              units=cube.units,
                              attributes=cube.attributes)

    for (coord, coord_dim) in shifted_dim_coords:
        if coord.name() not in coord_names:
            out_cube.add_dim_coord(coord, coord_dim[-1])

    # add back in coordinates not spanning the dimensions that were interpolated over
    for coord, coord_dim in coords_to_keep:
        # scalar coords
        if len(coord_dim) == 0:
            out_cube.add_aux_coord(coord, None)
        elif len(coord_dim) == 1:
            old_dim = cube.coord_dims(coord)
            idx = [el[1][0] == coord_dim for el in shifted_dim_coords].index(True)
            new_dim = shifted_dim_coords[idx][1][1]
            out_cube.add_aux_coord(coord, new_dim)
        else:
            pass

    # add back in the coordinates spanning the dimensions that were interpolated over
    for coord, coord_dim in coords_to_remove:
        if coord.name() in coord_names:
            # this was one of the specified coordinates
            idx = [el[0].name() == coord.name() for el in sample_points].index(True)
            coord_points = sample_points[idx][1]
            target_dim = len(out_cube.shape) - 1
        else:
            spanned_dimensions = []
            dim_points = []
            target_points = []
            for crd in cube.dim_coords:
                if len(set(cube.coord_dims(crd)).intersection(set(coord_dim))) > 0:
                    spanned_dimensions.append((crd.name(), cube.coord_dims(crd)))
                    dim_points.append(crd.points)
                    target_points.append(crd.points)
            f = RegularGridInterpolator(dim_points, coord.points, method="linear",
                                        **kwargs)
            shape_idx = []
            for name, dim in spanned_dimensions:
                shape_idx.extend(dim)
            shape_idx = tuple(shape_idx)
            arr_shape = [cube.shape[i] for i in shape_idx]
            target_dim = []
            set_dims = set([el[0].name() for el in shifted_dim_coords])
            set_span = set([el[0] for el in spanned_dimensions])
            for crd, (old_dim, new_dim) in shifted_dim_coords:
                if crd.name() in [el[0] for el in spanned_dimensions] and crd.name() not in coord_names:
                    target_dim.append(new_dim)
            related_index = []
            for sample_coord, sample_values in sample_points:
                check = [sample_coord.name() == el[0] for el in spanned_dimensions]
                if any(check):
                    idx = check.index(True)
                    related_index.append(idx)
                    target_points[idx] = sample_values
                    arr_shape[idx] = len(sample_values)
                    for crd, (old_dim, new_dim) in shifted_dim_coords:
                        if crd.name() == sample_coord.name():
                            target_dim.extend(new_dim)
            target_dim.append(len(out_cube.shape) - 1)
            combinations = _product(*target_points, related_index=tuple(related_index))
            coord_points = f(combinations)
            if len(set_dims.intersection(set_span)) > 0:
                coord_points = coord_points.reshape(arr_shape)

        new_coord = AuxCoord(coord_points,
                             standard_name=coord.standard_name,
                             long_name=coord.long_name,
                             var_name=coord.var_name,
                             units=coord.units,
                             attributes=coord.attributes)
        out_cube.add_aux_coord(new_coord, target_dim)

    # add back in the aux factories
    for fact, fact_dim in factories_to_remove:
        coord_mapping = {}
        for coord in fact.dependencies.values():
            coord_mapping[id(coord)] = out_cube.coord(coord.name())
        out_cube.add_aux_factory(fact.updated(coord_mapping))

    return out_cube


if __name__ == "__main__":
    cube = create_cube()
    print(cube)

    trajectory = np.array([np.array((-50 + i, -50 + i)) for i in range(3)])
#    sample_points = [("longitude", trajectory[:, 0]), ("latitude", trajectory[:, 1]), ("model_level_number", [6, 7, 8])]
    sample_points = [("longitude", trajectory[:, 0]), ("latitude", trajectory[:, 1])]

    start = time.time()
    traj_cube = _interpolate(cube, sample_points)
    print("_interpolate: {}s".format(time.time() - start))
    print(traj_cube)

    start = time.time()
    traj_cube_iris = iris.analysis.trajectory.interpolate(cube, sample_points,
                                                          method="linear")
    print("iris.analysis.trajectory.interpolate: {}s".format(time.time() - start))
    print(traj_cube_iris)

#    start = time.time()
#    traj_cube_iris = iris.analysis.trajectory.interpolate(cube, sample_points,
#                                                          method="nearest")
#    print("iris.analysis.trajectory.interpolate: {}s".format(time.time() - start))
#    print(traj_cube_iris)

    print("are results the same? {}".format(np.allclose(traj_cube.data, traj_cube_iris.data)))
#    print(traj_cube.data)
#    print(traj_cube_iris.data)

    for coord1, coord2 in zip(traj_cube.coords(), traj_cube_iris.coords()):
        print(coord1, coord2)
