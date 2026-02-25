import os

ROOT_DIR = "PMLR"          # đổi nếu cần
JSON_DIR = os.path.join(ROOT_DIR, "json")
PDF_DIR  = os.path.join(ROOT_DIR, "pdfs")

deleted_json = 0
deleted_pdf  = 0

print("🔍 Checking JSON directory...")

for fname in os.listdir(JSON_DIR):
    json_path = os.path.join(JSON_DIR, fname)

    if not os.path.isfile(json_path):
        continue

    # Nếu KHÔNG phải file .json
    if not fname.lower().endswith(".json"):
        base_name = os.path.splitext(fname)[0]

        # Xóa file "json" sai định dạng
        os.remove(json_path)
        deleted_json += 1
        print(f"❌ Deleted non-json file: {fname}")

        # Xóa PDF cùng tên (nếu tồn tại)
        pdf_path = os.path.join(PDF_DIR, base_name + ".pdf")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            deleted_pdf += 1
            print(f"   ↳ Deleted matching PDF: {base_name}.pdf")

print("\n✅ DONE")
print(f"   Deleted non-json files : {deleted_json}")
print(f"   Deleted matching PDFs  : {deleted_pdf}")