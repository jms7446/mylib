#!/usr/bin/env python

from unittest import TestCase, main

import common_tools as cmt


class CommonToolsTest(TestCase):

    def test_strip_margin(self):
        in_str = """a b
                   |c d 
                   |e f """
        actual = cmt.strip_margin(in_str)
        expected = "a b\nc d \ne f "
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    main()
