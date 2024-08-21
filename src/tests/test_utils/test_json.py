import json

import numpy as np
import pytest
from datetime import datetime
import pytz
from mlopus.utils import json_utils
from tzlocal import get_localzone


class Offsets:
    local_tz = pytz.timezone(get_localzone().key)
    utc_plus_2 = pytz.FixedOffset(+2 * 60)


class Times:
    noon = datetime(2024, 1, 1, 12)
    noon_in_utc = noon.astimezone(pytz.utc)
    noon_in_berlin = Offsets.utc_plus_2.localize(noon)
    now_here = Offsets.local_tz.localize(datetime.now())


class TestDumps:
    @pytest.mark.parametrize(
        "data, parser",
        [
            ([Times.noon_in_utc, Times.noon_in_berlin, Times.now_here], datetime.fromisoformat),
        ],
    )
    def test_preserve_data(self, data, parser):
        roundtrip = json.loads(json_utils.dumps(data))
        if parser:
            roundtrip = [parser(x) for x in roundtrip]
        assert roundtrip == data

    @pytest.mark.parametrize(
        "data",
        [
            [Times.noon_in_utc, Times.noon_in_berlin, Times.now_here],
        ],
    )
    def test_preserve_sorting(self, data):
        roundtrip = json.loads(json_utils.dumps(data))
        assert np.argsort(roundtrip).tolist() == np.argsort(data).tolist()
