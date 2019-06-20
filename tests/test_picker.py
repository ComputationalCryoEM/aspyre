import pytest
import numpy as np
from numpy.testing import assert_array_equal as aae
from aspyre.apple.picking import Picker


@pytest.fixture
def picker():
    '''
    A standard picker
    '''
    return Picker(
        particle_size=78,
        max_size=156,
        min_size=19,
        query_size=52,
        tau1=710,
        tau2=7100,
        moa=7,
        container_size=450,
        filename='test',
        output_directory='out_dir'
    )


def test_init(picker):
    assert picker.particle_size == 39
    assert picker.max_size == 78
    assert picker.min_size == 9
    assert picker.query_size == 26
    assert picker.tau1 == 710
    assert picker.tau2 == 7100
    assert picker.moa == 3
    assert picker.container_size == 225
    assert picker.filename == 'test'
    assert picker.output_directory == 'out_dir'

    # Picker with odd numbers for parameters that are truncated
    odd_picker = Picker(
        particle_size=79,
        max_size=157,
        min_size=20,
        query_size=53,
        tau1=710,
        tau2=7100,
        moa=7,
        container_size=451,
        filename='test',
        output_directory='out_dir'
    )

    assert odd_picker.particle_size == 39
    assert odd_picker.max_size == 78
    assert odd_picker.min_size == 10
    assert odd_picker.query_size == 26
    assert odd_picker.moa == 3
    assert odd_picker.container_size == 225
    assert odd_picker.tau1 == 710
    assert odd_picker.tau2 == 7100
    assert odd_picker.filename == 'test'
    assert odd_picker.output_directory == 'out_dir'


def test_read_mrc(picker, mocker):
    # don't want the config to affect these
    mocker.patch('src.aspyre.apple.picking.config.apple.mrc.margin_top', 1)
    mocker.patch('src.aspyre.apple.picking.config.apple.mrc.margin_bottom', 2)
    mocker.patch('src.aspyre.apple.picking.config.apple.mrc.margin_left', 3)
    mocker.patch('src.aspyre.apple.picking.config.apple.mrc.margin_right', 4)

    mocker.patch('src.aspyre.apple.picking.config.apple.mrc.shrink_factor', 5)
    # don't want to test these
    mock_resize = mocker.patch('src.aspyre.apple.picking.misc.imresize',
                               side_effect=lambda x, _, mode, interp: x)
    mock_corr = mocker.patch('src.aspyre.apple.picking.signal.correlate',
                             side_effect=lambda x, _, _2: x)

    # use a standard input
    mock_mrc = mocker.MagicMock()
    mock_mrc.__enter__().data = np.ones((16, 10))
    mock_mrc_open = mocker.patch('src.aspyre.apple.picking.mrcfile.open',
                                 return_value=mock_mrc)

    result = picker.read_mrc()
    mock_mrc_open.assert_called_once_with('test',
                                          mode='r+',
                                          permissive=True)
    mock_resize.assert_called_once_with(mocker.ANY,
                                        1/5,
                                        mode='F',
                                        interp='cubic')
    mock_corr.assert_called_once_with(mocker.ANY,
                                      mocker.ANY,
                                      'same')

    # now effectively testing size changes, should be min(16 - 3, 10 - 7) = 3x3
    aae(result, np.ones((3, 3)))
