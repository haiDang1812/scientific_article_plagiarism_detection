# extract_and_merge_text
import fitz
import os
import re


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def merge_blocks(blocks, y_threshold=10):
    blocks = sorted(blocks, key=lambda b: (b["page"], b["bbox"][1]))

    merged = []
    current = None

    for b in blocks:
        if current is None:
            current = b
            continue

        same_page = b["page"] == current["page"]
        close_y = abs(b["bbox"][1] - current["bbox"][3]) < y_threshold

        if same_page and close_y:
            current["text"] += " " + b["text"]
            current["bbox"][2] = max(current["bbox"][2], b["bbox"][2])
            current["bbox"][3] = b["bbox"][3]
        else:
            merged.append(current)
            current = b

    if current:
        merged.append(current)

    return merged


def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []

    for page_idx, page in enumerate(doc):
        raw_blocks = page.get_text("blocks")

        for block in raw_blocks:
            x0, y0, x1, y1, text, *_ = block

            if not text.strip():
                continue

            text = clean_text(text)

            blocks.append({
                "type": "text",
                "page": page_idx,
                "text": text,
                "bbox": [x0, y0, x1, y1]
            })

    merged_blocks = merge_blocks(blocks)

    return merged_blocks


# run
if __name__ == "__main__":
    import json

    base_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(
        base_dir,
        "..",
        "ijcai_paper",
        "2025_Q-MiniSAM2 A Quantization-based Benchmark for Resource-Efficient Video Segmentation.pdf"
    )

    result = process_pdf(pdf_path)

    output_dir = os.path.join(base_dir, "output_sample")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "output_clean.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Saved to:", output_path)