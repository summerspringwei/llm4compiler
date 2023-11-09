import unittest

from arm_assembly_regular_expression import get_hex16, get_extract_hex16, get_label


class TestRegularExpression(unittest.TestCase):

    def test_get_hex16(self):
        inst = "0000000000400808 <main>:\n"
        self.assertEqual(get_hex16(inst), ["0000000000400808"])
        self.assertEqual(get_hex16("0000000000400808"), ["0000000000400808"])
        self.assertEqual(get_hex16("000000000040080"), [])
    
    def test_exact_hex16_pattern(self):
        self.assertEqual(get_extract_hex16("0000000000400808"), ["0000000000400808"])
        self.assertEqual(get_extract_hex16("0000000000400808 main"), [])
        self.assertEqual(get_extract_hex16("000000000040080"), [])

    def test_get_label(self):
        self.assertEqual(get_label("0000000000400808 <main>:\n"), ["main"])
        self.assertEqual(get_label("0000000000400808 <main>\n"), [])
        self.assertEqual(get_label("<main>:\n"), ["main"])

if __name__ == '__main__':
    unittest.main()
