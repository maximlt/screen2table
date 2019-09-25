"""Tests for the helpers.

Do not cover a lot of (very few actually) corner cases.
But it's going to be useful for maintaining the app.
"""

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from screen2table import helpers


def test_functional_culvert():
    sd = helpers.ScreenData(
        [797, 692, 882, 938, 858, 1011],
        [285, 520, 621, 420, 428, 226],
        'culvert'
    )
    assert sd.nb_points == 6
    sd.process_screen_culvert()
    assert_almost_equal(
        sd.xz,
        np.array([
            [797., 285.],
            [692., 520.],
            [882., 621.],
            [938., 420.],
            [858., 428.],
            [1011., 226.]
        ])
    )
    assert_almost_equal(
        sd.xz_to_plot,
        np.array([
            [797.,  336.],
            [692.,  101.],
            [882.,    0.],
            [938.,  201.],
            [858.,  193.],
            [1011.,  395.],
            [797.,  336.]
        ])
    )
    assert_almost_equal(
        sd.xzinterp,
        np.array([
            [797., 336.],
            [736.68085106, 201.],
            [733.10638298, 193.],
            [692., 101.],
            [882., 0.],
            [910.13930348, 101.],
            [935.77114428, 193.],
            [938., 201.],
            [858., 193.],
            [864.05940594, 201.],
            [966.31188119, 336.],
            [1011., 395.],
            [797., 336.]
        ])
    )

    userparam_dict = {
        'minx': 1,
        'maxx': 2,
        'minz': 5,
        'maxz': 10,
        'angle': 0
    }
    up = helpers.UserParam(userparam_dict)
    up.validate()
    up_dict = up.convert_dict
    zw_real, zw_plot = helpers.polygon_to_levelwidth_table(
        sd.xzinterp,
        up_dict,
    )
    assert_almost_equal(
        zw_real,
        np.array([
            [5.000000000000000000e+00, 0.000000000000000000e+00],
            [6.278481012658227556e+00, 6.838222679704921703e-01],
            [7.443037974683544888e+00, 6.353127313475972482e-01],
            [7.544303797468354666e+00, 3.993058146607029180e-01],
            [9.253164556962024889e+00, 5.307582482386170586e-01],
            [1.000000000000000000e+01, 0.000000000000000000e+00]
        ])
    )
    assert_almost_equal(
        zw_plot,
        np.array([
            [5.000000000000000000e+00, 0.000000000000000000e+00],
            [6.278481012658227556e+00, -3.419111339852460851e-01],
            [7.443037974683544888e+00, -3.176563656737986241e-01],
            [7.544303797468354666e+00, -1.996529073303514590e-01],
            [9.253164556962024889e+00, -2.653791241193085293e-01],
            [1.000000000000000000e+01, 0.000000000000000000e+00],
            [9.253164556962024889e+00, 2.653791241193085293e-01],
            [7.544303797468354666e+00, 1.996529073303514590e-01],
            [7.443037974683544888e+00, 3.176563656737986241e-01],
            [6.278481012658227556e+00, 3.419111339852460851e-01],
            [5.000000000000000000e+00, 0.000000000000000000e+00]

        ])
    )
    assert helpers.calc_area(zw_real) == 2.250486091821753


def test_functional_xs():
    sd = helpers.ScreenData(
        [473, 1009, 1551],
        [154, 878, 457],
        'xs'
    )
    assert sd.nb_points == 3
    sd.process_screen_xs()
    userparam_dict = {
        'minx': 1,
        'maxx': 2,
        'minz': 5,
        'maxz': 10,
        'angle': 0
    }
    up = helpers.UserParam(userparam_dict)
    up.validate()
    up_dict = up.convert_dict
    xs_xz = helpers.scale_to_realdim(sd.xz, up_dict)
    assert_almost_equal(
        xs_xz,
        np.array([
            [1.000000000000000000e+00, 1.000000000000000000e+01],
            [1.497217068645640081e+00, 5.000000000000000000e+00],
            [2.000000000000000000e+00, 7.907458563535911367e+00]
        ])
    )
    assert helpers.calc_length(xs_xz) == 7.975272780439953


def test_detect_two_lines_intersection():
    assert helpers.ScreenData.detect_two_lines_intersection(
        [0, 0], [1, 0], [0, 1], [1, 1]
    ) is False
    assert helpers.ScreenData.detect_two_lines_intersection(
        [0, 0], [1, 1], [0, 1], [1, 1]
    ) is False
    assert helpers.ScreenData.detect_two_lines_intersection(
        [0, 0], [1, 1], [0, 1], [1, 0]
    ) is True


def test_is_self_intersecting_polyg():
    sc = helpers.ScreenData(
        [0, -0.5, 0],
        [0, 0.5, 1],
        'culvert'
    )
    assert sc.is_self_intersecting_polyg() is False
    sc = helpers.ScreenData(
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        'culvert'
    )
    assert sc.is_self_intersecting_polyg() is True


def test_process_screen_culvert():
    sc = helpers.ScreenData(
        [0],
        [0],
        'culvert'
    )
    sc.process_screen_culvert()
    assert sc.is_ok is False
    assert sc.dict_tkwarn['toofewpoints']['is_error'] is True
    sc = helpers.ScreenData(
        [0, 1],
        [0, 1],
        'culvert'
    )
    sc.process_screen_culvert()
    assert sc.is_ok is False
    assert sc.dict_tkwarn['toofewpoints']['is_error'] is True
    sc = helpers.ScreenData(
        [1, 0, 1, 1],
        [0, 0.5, 1, 0],
        'culvert'
    )
    sc.process_screen_culvert()
    assert sc.is_ok is False
    assert sc.dict_tkwarn['duplicates']['is_error'] is True
    sc = helpers.ScreenData(
        [1, 0, 0, 1],
        [0, 0.5, 0.5, 1],
        'culvert'
    )
    sc.process_screen_culvert()
    assert sc.is_ok is False
    assert sc.dict_tkwarn['duplicates']['is_error'] is True
    sc = helpers.ScreenData(
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        'culvert'
    )
    sc.process_screen_culvert()
    assert sc.is_ok is False
    assert sc.dict_tkwarn['self_intersection']['is_error'] is True
    sc = helpers.ScreenData(
        [0, -1, 0],
        [0, -0.5, -1],
        'culvert'
    )
    sc.process_screen_culvert()
    assert sc.is_ok is True
    assert all(val['is_error'] is False for val in sc.dict_tkwarn.values())
    assert sc.nb_points == 3
    assert_array_equal(
        sc.xz,
        np.array([
            [0, 0],
            [-1, -0.5],
            [0, -1]
        ])
    )
    assert_array_equal(
        sc.xz_to_plot,
        np.array([
            [0, 0],
            [-1, 0.5],
            [0, 1],
            [0, 0]
        ])
    )
    assert_array_equal(
        sc.xzinterp,
        np.array([
            [0, 0],
            [-1, 0.5],
            [0, 1],
            [0, 0.5],
            [0, 0]
        ])
    )


def test_process_screen_xs():
    sc = helpers.ScreenData(
        [0],
        [0],
        'xs'
    )
    sc.process_screen_xs()
    assert sc.is_ok is False
    assert sc.dict_tkwarn['toofewpoints']['is_error'] is True
    sc = helpers.ScreenData(
        [1, 0, 1, 1],
        [0, 0.5, 1, 0],
        'xs'
    )
    sc.process_screen_xs()
    assert sc.is_ok is False
    assert sc.dict_tkwarn['duplicates']['is_error'] is True
    sc = helpers.ScreenData(
        [0, -1, 0],
        [0, -0.5, -1],
        'xs'
    )
    sc.process_screen_xs()
    assert sc.is_ok is True
    assert all(val['is_error'] is False for val in sc.dict_tkwarn.values())
    assert sc.xzinterp is None
    assert sc.nb_points == 3
    assert_array_equal(sc.xz, sc.xz_to_plot)
    assert_array_equal(
        sc.xz_to_plot,
        np.array([
            [0, 0],
            [-1, 0.5],
            [0, 1],
        ])
    )


def test_arr_contain_duplicates():
    arr = np.array([
        [0, 0],
        [-0.5, 0.5],
        [0, 1],
    ])
    assert helpers.arr_contain_duplicates(arr) is False
    arr = np.array([
        [0, 0],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [0, 1],
    ])
    assert helpers.arr_contain_duplicates(arr) is True


def test_flip():
    assert_array_equal(
        helpers.flip(np.array([[0, 4], [1, 3],  [2, 2]]), 0),
        np.array([[2, 4], [1, 3], [0, 2]])
    )
    assert_array_equal(
        helpers.flip(np.array([[0, 4], [1, 3],  [2, 2]]), 1),
        np.array([[0, 0], [1, 1], [2, 2]])
    )


def test_are_all_z_unique():
    pg = np.array([
        [0, 0],
        [-0.5, 0.5],
        [0, 1],
        [0, 0]
    ])
    assert helpers.are_all_z_unique(pg) is True
    pg = np.array([
        [0, 0],
        [-0.5, 0.5],
        [0, 1],
    ])
    assert helpers.are_all_z_unique(pg) is True
    pg = np.array([
        [0, 0],
        [-0.5, 0.5],
        [0, 0.5],
        [0, 0]
    ])
    assert helpers.are_all_z_unique(pg) is False


def test_modify_equal_z():
    np.random.seed(13)
    xz = np.array([
        [0, 0],
        [-1, 0.5],
        [1, 1],
        [1, 0.5]
    ])
    assert_almost_equal(
        helpers.modify_equal_z(xz),
        np.array([
            [0.,  0.],
            [-1.,  0.50055485],
            [1.,  1.],
            [1.,  0.49947561]
        ])
    )


def test_add_z_points_to_polygon():
    pg = np.array([
        [0, 0],
        [-0.5, 0.5],
        [0, 1],
        [0, 0]
    ])
    assert_array_equal(
        helpers.add_z_points_to_polygon(pg),
        np.array([
            [0, 0],
            [-0.5, 0.5],
            [0, 1],
            [0, 0.5],
            [0, 0]
        ])
    )


def test_linear_interp_pt():
    assert helpers.linear_interp_pt(0.5, 0, 1, 0, 1) == 0.5
    assert helpers.linear_interp_pt(-0.5, 0, -1, 0, -1) == -0.5


def test_UserParam():
    up = helpers.UserParam(
        {
            'minx': 0,
            'maxx': 1,
            'minz': 10,
            'maxz': 20,
            'angle': 15
        }
    )
    rule_dict = up.validate()
    for rule in rule_dict.values():
        assert rule.is_ok is True
    for param_value in up.convert_dict.values():
        assert isinstance(param_value, float)
    up = helpers.UserParam(
        {
            'minx': 1,
            'maxx': 0,
            'minz': 20,
            'maxz': 10,
            'angle': 200
        }
    )
    rule_dict = up.validate()
    for k, rule in rule_dict.items():
        if k != 'float':
            assert rule.is_ok is False
    for param_value in up.convert_dict.values():
        assert isinstance(param_value, float)
    up = helpers.UserParam(
        {
            'minx': 'a',
            'maxx': 1,
            'minz': 10,
            'maxz': 20,
            'angle': 15
        }
    )
    rule_dict = up.validate()
    assert rule_dict['float'].is_ok is False
    assert up.convert_dict is None


def test_polygon_to_levelwidth_table():
    pass


def test_polygon_to_heightwidth_table():
    pg = np.array([
        [0, 0],
        [-0.5, 0.5],
        [0, 1],
        [0.5, 0.5],
        [0, 0]
    ])
    assert_array_equal(
        helpers.polygon_to_heightwidth_table(pg),
        np.array([
            [0., 0.],
            [0.5, 1],
            [1., 0.],
        ])
    )
    pg = np.array([
        [0, 0],
        [-0.5, 0.5],
        [0, 1],
        [0.5, 0.5]
    ])
    assert_array_equal(
        helpers.polygon_to_heightwidth_table(pg),
        np.array([
            [0., 0.],
            [0.5, 1],
            [1., 0.],
        ])
    )


def test_scale_to_realdim():
    up = {
        'minx': 0,
        'maxx': 1,
        'minz': 10,
        'maxz': 20,
        'angle': 15,
    }
    pg = np.array([
        [868., 0.],
        [790.51162791, 168.],
        [749., 258.],
        [978., 168.],
        [868., 0.]
    ])
    assert_almost_equal(
        helpers.scale_to_realdim(pg, up),
        np.array([
            [0.50194399, 10.],
            [0.17509674, 16.51162791],
            [0., 20.],
            [0.96592583, 16.51162791],
            [0.50194399, 10.]
        ])
    )


def test_zw_to_plot():
    zw = np.array([[0, 0], [0.5, 1], [1, 0]])
    assert_array_equal(
        helpers.zw_to_plot(zw),
        np.array([
            [0.,  0.],
            [0.5, -0.5],
            [1.,  0.],
            [0.5,  0.5],
            [0.,  0.]
        ])
    )


def test_remove_close_points():
    zw = np.array([[0, 0], [0.05, 0.05]])
    assert_array_equal(
        helpers.remove_close_points(zw, threshold=0.1),
        np.array([[0, 0]])
    )
    zw = np.array([[0, 0], [0.05, 0.05], [0.06, 0.06], [0.16, 0.16]])
    assert_array_equal(
        helpers.remove_close_points(zw, threshold=0.1),
        np.array([[0, 0], [0.16, 0.16]])
    )
    zw = np.array([[0, 0], [0.05, 0.05], [0.995, 0.995], [1, 1]])
    assert_array_equal(
        helpers.remove_close_points(zw, threshold=0.1),
        np.array([[0, 0], [0.995, 0.995]])
    )
    zw = np.array([[0, 0], [0.05, 0.1]])
    assert_array_equal(
        helpers.remove_close_points(zw, threshold=0.1),
        np.array([[0, 0], [0.05, 0.1]])
    )
    zw = np.array([[0, 0], [0.1, 0.05]])
    assert_array_equal(
        helpers.remove_close_points(zw, threshold=0.1),
        np.array([[0, 0], [0.1, 0.05]])
    )


def test_calc_length():
    pl = np.array([[0, 0], [0, 1]])
    assert helpers.calc_length(pl) == 1
    pl = np.array([[0, 0], [0, 1], [0, 0]])
    assert helpers.calc_length(pl) == 2
    pl = np.array([[0, 0], [0, 0]])
    assert helpers.calc_length(pl) == 0


def test_calc_area():
    pg = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    assert helpers.calc_area(pg) == 1
    pg = np.array([[0, 1], [1, 1]])
    assert helpers.calc_area(pg) == 1


def test_to_clipboard_for_excel():
    from win32 import win32clipboard
    data = np.array([[0, 0], [1, 1]])
    helpers.to_clipboard_for_excel(data, decimals=1)
    win32clipboard.OpenClipboard()
    text = win32clipboard.GetClipboardData()
    assert text == '0.0\t0.0\r\n1.0\t1.0'
