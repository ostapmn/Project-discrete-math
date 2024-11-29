"""Module for testing lib functions"""

import re
import unicodedata
from pdfminer.high_level import extract_text

PATTERN = re.compile(r'\[\d*?\]')

def normalize_text(text):
    normalized = unicodedata.normalize("NFKD", text)
    return normalized

x = extract_text('test_papers/higherandderivedstacksaglobaloverview.pdf')

print(normalize_text(x))

# print(x.pages[1])
# text = ''
# marker = False
# for page in x.pages:
#     words = page.extract_text()
#     if ('References' in words) or marker:
#         marker = True
#         text += words

