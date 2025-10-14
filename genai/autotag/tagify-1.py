#!/usr/bin/env python3
import csv
import sys

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, ArrayObject, NumberObject, NameObject, TextStringObject


def main():
    pdf_file = sys.argv[1]
    csv_file = sys.argv[2]

    # Load PDF
    reader = PdfReader(pdf_file)
    writer = PdfWriter()

    # Copy all pages to writer
    for page in reader.pages:
        writer.add_page(page)

    # Remove existing form fields
    if '/AcroForm' in writer._root_object:
        del writer._root_object['/AcroForm']

    # Load CSV
    fields = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            # Strip whitespace from keys and values
            cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
            fields.append(cleaned_row)

    # Create AcroForm dictionary
    acroform = DictionaryObject()
    acroform.update({
        NameObject('/Fields'): ArrayObject(),
        NameObject('/NeedAppearances'): NameObject('/true')
    })

    # Create new fields from CSV
    for field_data in fields:
        name = field_data['Name']
        page_num = int(field_data['Page']) - 1  # 0-indexed
        heading = field_data['Heading']
        section = field_data['Section']
        x = float(field_data['X'])
        y = float(field_data['Y'])
        width = float(field_data['Width'])
        height = float(field_data['Height'])

        # Create field identifier
        field_id = f"{page_num + 1}>{heading}>{section}>{name}"

        # Create field annotation
        field = DictionaryObject()
        field.update({
            NameObject('/FT'): NameObject('/Tx'),  # Text field
            NameObject('/T'): TextStringObject(field_id),
            NameObject('/V'): TextStringObject(''),
            NameObject('/Rect'): ArrayObject([
                NumberObject(x),
                NumberObject(y),
                NumberObject(x + width),
                NumberObject(y + height)
            ]),
            NameObject('/Type'): NameObject('/Annot'),
            NameObject('/Subtype'): NameObject('/Widget'),
            NameObject('/P'): writer.pages[page_num].indirect_reference
        })

        # Add field to AcroForm
        acroform[NameObject('/Fields')].append(writer._add_object(field))

        # Add annotation to page
        if '/Annots' not in writer.pages[page_num]:
            writer.pages[page_num][NameObject('/Annots')] = ArrayObject()
        writer.pages[page_num]['/Annots'].append(field.indirect_reference)

    # Add AcroForm to PDF
    writer._root_object.update({
        NameObject('/AcroForm'): acroform
    })

    # Write output
    output_file = pdf_file.replace('.pdf', '_with_fields.pdf')
    with open(output_file, 'wb') as f:
        writer.write(f)

    print(f"Created {output_file} with {len(fields)} fields")


if __name__ == '__main__':
    main()
