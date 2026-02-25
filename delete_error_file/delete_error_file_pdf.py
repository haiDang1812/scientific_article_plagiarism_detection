import os

PDF_DIR = "PMLR/pdfs"   # đổi nếu cần

deleted = 0

print("🔍 Checking PDF directory...")

for fname in os.listdir(PDF_DIR):
    fpath = os.path.join(PDF_DIR, fname)

    if not os.path.isfile(fpath):
        continue

    # Nếu KHÔNG phải file PDF
    if not fname.lower().endswith(".pdf"):
        os.remove(fpath)
        deleted += 1
        print(f"❌ Deleted non-pdf file: {fname}")

print("\n✅ DONE")
print(f"   Deleted files: {deleted}")