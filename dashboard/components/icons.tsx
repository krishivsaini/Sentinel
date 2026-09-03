// SVG icons ported from the design mockup. All inherit `currentColor` + accept a size.
type P = { size?: number; className?: string };
const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
});

export const GridIcon = ({ size = 15, className }: P) => (
  <svg {...base(size)} className={className}>
    <rect x="3" y="3" width="7" height="7" rx="1.5" />
    <rect x="14" y="3" width="7" height="7" rx="1.5" />
    <rect x="3" y="14" width="7" height="7" rx="1.5" />
    <rect x="14" y="14" width="7" height="7" rx="1.5" />
  </svg>
);

export const ListIcon = ({ size = 15, className }: P) => (
  <svg {...base(size)} className={className}>
    <line x1="8" y1="6" x2="20" y2="6" />
    <line x1="8" y1="12" x2="20" y2="12" />
    <line x1="8" y1="18" x2="20" y2="18" />
    <circle cx="4" cy="6" r="1.2" fill="currentColor" stroke="none" />
    <circle cx="4" cy="12" r="1.2" fill="currentColor" stroke="none" />
    <circle cx="4" cy="18" r="1.2" fill="currentColor" stroke="none" />
  </svg>
);

export const SearchIcon = ({ size = 15, className }: P) => (
  <svg {...base(size)} className={className}>
    <circle cx="11" cy="11" r="7" />
    <line x1="16" y1="16" x2="21" y2="21" />
  </svg>
);

export const AttrIcon = ({ size = 15, className }: P) => (
  <svg {...base(size)} className={className}>
    <path d="M12 3v18" />
    <rect x="3.5" y="8" width="6" height="8" rx="1.2" />
    <rect x="14.5" y="5" width="6" height="11" rx="1.2" />
  </svg>
);

export const QueryIcon = ({ size = 15, className }: P) => (
  <svg {...base(size)} className={className}>
    <path d="M4 5h16v11H9l-5 4z" strokeLinejoin="round" />
  </svg>
);

export const ShieldIcon = ({ size = 15, className }: P) => (
  <svg {...base(size)} className={className}>
    <path d="M12 3l7 3v5c0 4.4-3 8-7 10-4-2-7-5.6-7-10V6z" strokeLinejoin="round" />
    <path d="M9 12l2 2 4-4" strokeLinejoin="round" />
  </svg>
);

export const SunIcon = ({ size = 13, className }: P) => (
  <svg {...base(size)} strokeWidth={1.9} className={className}>
    <circle cx="12" cy="12" r="4.2" />
    <line x1="12" y1="2.5" x2="12" y2="5" />
    <line x1="12" y1="19" x2="12" y2="21.5" />
    <line x1="2.5" y1="12" x2="5" y2="12" />
    <line x1="19" y1="12" x2="21.5" y2="12" />
    <line x1="5.2" y1="5.2" x2="6.9" y2="6.9" />
    <line x1="17.1" y1="17.1" x2="18.8" y2="18.8" />
    <line x1="5.2" y1="18.8" x2="6.9" y2="17.1" />
    <line x1="17.1" y1="6.9" x2="18.8" y2="5.2" />
  </svg>
);

export const MoonIcon = ({ size = 13, className }: P) => (
  <svg {...base(size)} strokeWidth={1.9} className={className}>
    <path d="M21 12.8A8 8 0 1111 3a6.2 6.2 0 0010 9.8z" strokeLinejoin="round" />
  </svg>
);

export const CheckIcon = ({ size = 13, className }: P) => (
  <svg {...base(size)} strokeWidth={2.2} className={className}>
    <path d="M5 13l4 4 10-10" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export const CaretIcon = ({ size = 12, className }: P) => (
  <svg {...base(size)} strokeWidth={2.4} className={className}>
    <path d="M9 6l6 6-6 6" />
  </svg>
);

export const ChevronDown = ({ size = 12, className }: P) => (
  <svg {...base(size)} strokeWidth={2} className={className}>
    <path d="M6 9l6 6 6-6" />
  </svg>
);

export const LogoMark = ({ size = 15 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="8.5" stroke="#fff" strokeWidth="1.7" />
    <circle cx="12" cy="12" r="2.4" fill="#818cf8" />
  </svg>
);
