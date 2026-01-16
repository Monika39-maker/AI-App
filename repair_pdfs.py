from pathlib import Path
import pikepdf

DATA_DIR = Path(__file__).parent / "AIdata"
for pdf in DATA_DIR.glob("*.pdf"):
    out = pdf.with_name(pdf.stem + "-repaired.pdf")
    try:
        print(f"Opening {pdf}")
        with pikepdf.open(pdf) as doc:
            doc.save(out)
        print(f"Saved repaired: {out}")
    except Exception as e:
        print(f"Failed to repair {pdf}: {e}")
