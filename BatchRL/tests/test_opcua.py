from unittest import TestCase

from opcua_empa.opcua_util import th_string_to_node_name


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
