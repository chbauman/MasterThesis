import datetime
from unittest import TestCase

from opcua_empa.opcua_util import th_string_to_node_name, get_min_diff


class TestOpcua(TestCase):
    """Tests the opcua client and related stuff.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_string_manipulation(self):
        inp = "Hoi_Du"
        exp = "strHoi.strDu"
        res = th_string_to_node_name(inp)
        self.assertEqual(res[-len(exp):], exp)

    def test_min_diff(self):
        d1 = datetime.datetime(2005, 7, 14, 13, 30)
        d2 = datetime.datetime(2005, 7, 14, 12, 30)
        min_diff = get_min_diff(d2, d1)
        self.assertAlmostEqual(min_diff, 60.0)
