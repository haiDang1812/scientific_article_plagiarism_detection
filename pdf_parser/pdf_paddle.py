# extract_text_images_and_ocr
import fitz
import os
import io
from PIL import Image
from paddleocr import PaddleOCR

OUTPUT_IMG_DIR = "images"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

ocr = PaddleOCR(use_angle_cls=True, lang="en")


def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    data = []

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block

            if text.strip():
                data.append({
                    "type": "text",
                    "page": page_idx,
                    "text": text.strip(),
                    "bbox": [x0, y0, x1, y1]
                })

        images = page.get_images(full=True)

        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            img_path = os.path.join(
                OUTPUT_IMG_DIR,
                f"page{page_idx}_img{img_idx}.png"
            )
            image.save(img_path)

            ocr_result = ocr.ocr(img_path)

            ocr_texts = []
            for line in ocr_result:
                for _, (txt, score) in line:
                    ocr_texts.append({
                        "text": txt,
                        "confidence": float(score)
                    })

            data.append({
                "type": "figure",
                "page": page_idx,
                "image_path": img_path,
                "ocr_text": ocr_texts
            })

    return data


# run
if __name__ == "__main__":
    result = process_pdf("paper.pdf")

    import json
    print(json.dumps(result[:10], indent=2, ensure_ascii=False))