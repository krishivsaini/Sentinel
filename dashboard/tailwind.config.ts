import type { Config } from "tailwindcss";

// Colors are CSS variables (defined in app/globals.css for light + dark) so the whole palette
// theme-switches by toggling `data-theme` on <html>. This mirrors the design mockup's token set.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        panel: {
          DEFAULT: "var(--panel)",
          2: "var(--panel-2)",
          3: "var(--panel-3)",
        },
        border: {
          DEFAULT: "var(--border)",
          2: "var(--border-2)",
          soft: "var(--border-soft)",
          hair: "var(--border-hair)",
        },
        text: {
          DEFAULT: "var(--text)",
          1: "var(--text-1)",
          2: "var(--text-2)",
          3: "var(--text-3)",
          4: "var(--text-4)",
          5: "var(--text-5)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          2: "var(--accent-2)",
          soft: "var(--accent-soft)",
        },
        chip: "var(--chip)",
        ok: { fg: "var(--ok-fg)", bg: "var(--ok-bg)", bd: "var(--ok-bd)" },
        bad: { fg: "var(--bad-fg)", bg: "var(--bad-bg)" },
        warn: { fg: "var(--warn-fg)", bg: "var(--warn-bg)" },
        ret: { fg: "var(--ret-fg)", bg: "var(--ret-bg)", solid: "var(--ret-solid)" },
        gen: { fg: "var(--gen-fg)", bg: "var(--gen-bg)", solid: "var(--gen-solid)" },
        mark: "var(--mark)",
        "input-bd": "var(--input-bd)",
        dash: "var(--dash)",
        grid: "var(--grid)",
        line: { faith: "var(--line-faith)", rel: "var(--line-rel)", rec: "var(--line-rec)" },
        cu: {
          bd: "var(--cu-bd)",
          bg: "var(--cu-bg)",
          "chip-bg": "var(--cu-chip-bg)",
          "chip-fg": "var(--cu-chip-fg)",
        },
        btn: { bg: "var(--btn-bg)", fg: "var(--btn-fg)" },
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
