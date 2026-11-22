"""Write data/guardrails/given_names.txt: given names from Wikidata (CC0), for PII name checks.

Entity recognition alone can't be trusted on technical text: on the RFC corpus spaCy's small
English model tagged JSON, MUST, Huffman and "Header Parameter" as people. sentinel.guardrails.pii
therefore keeps a PERSON span only when it also looks like a person's name, and one of the ways
it can is by starting with a known given name. The list comes from Wikidata's male, female and
unisex given-name items (any language's English or multilingual label, on at least two
Wikipedias) rather than a national registry, so the check doesn't protect "Daniel" better than
"Kwame" or "Wei".

    python scripts/fetch_given_names.py
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

OUT = Path(__file__).resolve().parents[1] / "data" / "guardrails" / "given_names.txt"
ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "Sentinel-RAG-data-fetcher/1.0 (+https://github.com/krishivsaini/Sentinel)"
QUERY = """
SELECT ?label WHERE {
  VALUES ?type { wd:Q12308941 wd:Q11879590 wd:Q3409032 }
  ?n wdt:P31 ?type ; wikibase:sitelinks ?s ; rdfs:label ?label .
  FILTER(?s >= 2)
  FILTER(LANG(?label) = "en" || LANG(?label) = "mul")
}
"""
# One word of letters, optionally with an inner apostrophe or hyphen (O'Neill-style is a surname
# matter; given names like Jean-Luc or D'Andre keep their joiner).
NAME_RE = re.compile(r"^[^\W\d_]+(?:['’-][^\W\d_]+)*$")


def main() -> int:
    url = f"{ENDPOINT}?{urlencode({'query': QUERY})}"
    proc = subprocess.run(["curl", "-sS", "--fail", "--max-time", "180", "-A", USER_AGENT,
                           "-H", "Accept: text/csv", url], capture_output=True, check=True)
    rows = csv.DictReader(io.StringIO(proc.stdout.decode("utf-8")))
    names = sorted({r["label"].strip() for r in rows if NAME_RE.match(r["label"].strip())},
                   key=str.casefold)
    if len(names) < 10_000:
        print(f"only {len(names)} names returned; refusing to overwrite {OUT.name}")
        return 1
    header = (f"# Given names from Wikidata (CC0): male/female/unisex given-name items on 2+ "
              f"Wikipedias.\n# Fetched {datetime.now(UTC):%Y-%m-%d} by "
              f"scripts/fetch_given_names.py; one per line.\n")
    OUT.write_text(header + "\n".join(names) + "\n", encoding="utf-8")
    print(f"wrote {len(names)} names to {OUT.relative_to(OUT.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
