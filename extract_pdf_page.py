#!/usr/bin/env python

import base64
import sys

from hwtt.file_handling import extract_images_from_pdf

pdf_path = sys.argv[1]
images = extract_images_from_pdf(pdf_path)
page = int(sys.argv[2])

print(base64.b64encode(images[page]).decode('ascii'))
