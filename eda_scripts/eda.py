import json
from collections import defaultdict
from pathlib import Path

JSON_DIR = Path(__file__).parent / "CVF_MAIN_FIXED" / "json"


def load_all(json_dir: Path):
    records = []
    for f in json_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_filename"] = f.name
            records.append(data)
        except Exception as e:
            print(f"  [WARN] Cannot read {f.name}: {e}")
    return records


def separator(char="=", width=60):
    print(char * width)


def main():
    records = load_all(JSON_DIR)

    separator()
    print("  CVF PAPER COLLECTION — EDA REPORT")
    separator()

    # ── 1. Tổng số file ──────────────────────────────────────
    total = len(records)
    print(f"\n  Tổng số file JSON   : {total:,}")

    # ── 2. Thống kê theo hội nghị ────────────────────────────
    by_conf = defaultdict(list)
    for r in records:
        conf = r.get("conference", "UNKNOWN")
        by_conf[conf].append(r)

    print(f"  Số hội nghị khác nhau: {len(by_conf)}\n")

    separator("-")
    print(f"  {'Hội nghị':<10}  {'Số paper':>10}  {'Tỉ lệ':>8}")
    separator("-")
    for conf in sorted(by_conf, key=lambda c: -len(by_conf[c])):
        count = len(by_conf[conf])
        pct = count / total * 100
        print(f"  {conf:<10}  {count:>10,}  {pct:>7.2f}%")
    separator("-")

    # ── 3. Thống kê từng hội nghị theo năm ──────────────────
    for conf in sorted(by_conf):
        papers = by_conf[conf]
        by_year = defaultdict(int)
        for p in papers:
            by_year[p.get("year", "??")] += 1

        print(f"\n  [{conf}]  —  {len(papers):,} papers tổng cộng")
        separator("-", 40)
        print(f"    {'Năm':<8}  {'Số paper':>10}")
        separator("-", 40)
        for year in sorted(by_year):
            print(f"    {str(year):<8}  {by_year[year]:>10,}")
        separator("-", 40)

    separator()
    print("  Kết thúc báo cáo.")
    separator()


if __name__ == "__main__":
    main()
