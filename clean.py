import re
import sqlite3
from contextlib import closing

import pandas as pd


def cpu_generation(t):
    m = re.search(r"i[3579][- ]?(\d{4,5})", t)
    if m:
        n = m.group(1)
        return int(n[:2]) if (len(n) == 5 or n.startswith("1")) else int(n[0])
    m = re.search(r"ryzen\s?[3579]\s?(?:pro\s?)?(\d)\d{3}", t)
    if m:
        return int(m.group(1))
    return None


def parse(title):
    t = title.lower()
    ram = re.search(r"(?<!\d)(\d{1,2})\s?gb\s?(?:ram|ddr|memory)?", t)
    ssd = re.search(r"(\d{3,4})\s?gb(?:\s?(?:ssd|nvme|emmc|hdd))?|(\d)\s?tb", t)
    intel_cpu = re.search(r"\b(i[3579])[- ]?(\d{4,5})([a-z]\d?)?\b", t)
    ryzen_cpu = re.search(r"\b(ryzen\s?[3579])\s?(?:pro\s?)?(\d{4,5})([a-z]\d?)?\b", t)
    apple_cpu = re.search(r"\b(m[1-4])\b", t)
    cpu_family = re.search(r"\b(i[3579])\b|\b(ryzen\s?[3579])\b|\b(m[1-4])\b", t)
    cpu = (
        intel_cpu.group(1) if intel_cpu else
        ryzen_cpu.group(1) if ryzen_cpu else
        apple_cpu.group(1) if apple_cpu else
        next((group for group in cpu_family.groups() if group), None) if cpu_family else None
    )
    cpu_model = (f"{intel_cpu.group(1)}-{intel_cpu.group(2)}{intel_cpu.group(3) or ''}" if intel_cpu else
                 f"{ryzen_cpu.group(1)} {ryzen_cpu.group(2)}{ryzen_cpu.group(3) or ''}" if ryzen_cpu else
                 apple_cpu.group(1) if apple_cpu else None)
    screen = re.search(r"(1[1-7](?:\.\d)?)\s?(?:\"|inch|in\b)", t)
    brand = next((b for b in ["apple", "macbook", "dell", "hp", "lenovo", "thinkpad",
                              "asus", "acer", "microsoft", "surface"] if b in t), None)
    return {
        "ram_gb": int(ram.group(1)) if ram else None,
        "storage_gb": (int(ssd.group(1)) if ssd and ssd.group(1)
                       else int(ssd.group(2)) * 1000 if ssd and ssd.group(2) else None),
        "cpu": cpu,
        "cpu_model": cpu_model,
        "cpu_gen": cpu_generation(t),
        "screen_in": float(screen.group(1)) if screen else None,
        "brand": brand,
        "not_laptop": bool(re.search(
            r"\b(?:desktop|tower|all[- ]in[- ]one|mini pc|thin client)\b", t)),
        "for_parts": bool(re.search(
            r"for parts|dead batt|\bdead\b|no power|bios lock|boots to bios|remote managed|"
            r"parts only|not working|broken|cracked|\bread\b|see desc|\bas[- ]is\b|"
            r"damaged|bad display|lcd (?:spot|damage|issue|bleed)|"
            r"logic ?/? ?(?:mother)? ?board|mother ?board|top case|topcase|battery only|screen only|"
            r"no ssd|no hdd|no storage|no ram|no battery|water damage|liquid damage|liquid spill|faulty|"
            r"defective|needs repair|for repair|swollen|icloud|activation lock|\bmdm\b|\blocked\b", t)),
    }


def build_clean():
    with closing(sqlite3.connect("data/prices.db")) as con, con:
        df = pd.read_sql("SELECT * FROM raw_listings", con)
        df = df[df.currency == "USD"]
        specs = df.title.apply(parse).apply(pd.Series)
        df = pd.concat([df, specs], axis=1)
        bad_condition = df.condition.fillna("").str.contains(
            r"for parts|not working|salvage", case=False, regex=True
        )
        df = df[~df.for_parts & ~df.not_laptop & ~bad_condition]
        df = df[df.price.between(100, 4000)]
        df = df.drop_duplicates(["snapshot_date", "title", "price"])
        df.to_sql("clean_listings", con, if_exists="replace", index=False)
    print(f"{len(df)} clean rows | missing RAM: {df.ram_gb.isna().mean():.0%}")


if __name__ == "__main__":
    build_clean()