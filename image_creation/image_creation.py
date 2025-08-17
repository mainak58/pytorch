from pdf2image import convert_from_path
import os
import tempfile

pdf_folder = r"C:\Users\MAINAK\Desktop\opencv\image_creation\pdfs"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

with tempfile.TemporaryDirectory() as path:
    counter = 1

    for pdf in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf)
        print(f"Processing: {pdf_path}")
        images_from_path = convert_from_path(
            pdf_path,
            poppler_path=r"C:\poppler\Library\bin",
            use_pdftocairo=True
        )

        for img in images_from_path:
            file_name = file_name = os.path.join(r"C:\Users\MAINAK\Desktop\opencv\image_creation\train",f"{counter}.png"
            )
            img.save(file_name, "PNG")
            print(f"Saved: {file_name}")
            counter += 1
