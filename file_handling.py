import pdfplumber


def extract_images_from_pdf(pdf_path):
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            images.append(page.images[0]['stream'].get_rawdata())
    return images
