// Copies the eval JSON exports (written by the Python pipeline to ../data) into public/data,
// where the Next app serves them. Resilient: if a source file is missing (e.g. a fresh Vercel
// clone where ../data is gitignored), it leaves the committed snapshot in public/data untouched.
import { existsSync, mkdirSync, copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "data");
const dst = join(here, "..", "public", "data");
mkdirSync(dst, { recursive: true });

const files = ["latest.json", "history.json", "latency_report.json"];
let copied = 0;
for (const f of files) {
  const from = join(src, f);
  if (existsSync(from)) {
    copyFileSync(from, join(dst, f));
    copied++;
  }
}
console.log(`sync-data: copied ${copied}/${files.length} file(s) into public/data`);
