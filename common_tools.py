#!/usr/bin/env python

import re


def strip_margin(text):
    return re.sub('\n[ \t]*\|', '\n', text)


def strip_heredoc(text):
    indent = len(min(re.findall('\n[ \t]*(?=\S)', text) or ['']))
    pattern = r'\n[ \t]{%d}' % (indent - 1)
    return re.sub(pattern, '\n', text)


