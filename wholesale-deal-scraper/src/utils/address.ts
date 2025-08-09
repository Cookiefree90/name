export type ParsedAddress = {
  raw: string;
  street?: string;
  city?: string;
  state?: string;
  zip?: string;
};

const CITY_STATE_RE = /\b([A-Za-z\s]+),\s*([A-Z]{2})\b/;
const ZIP_RE = /\b(\d{5})(?:-\d{4})?\b/;

export function parseAddress(text?: string, fallbackCity?: string, fallbackState?: string): ParsedAddress | undefined {
  if (!text) return undefined;
  const trimmed = text.replace(/\s+/g, ' ').trim();
  if (!trimmed) return undefined;

  const zip = trimmed.match(ZIP_RE)?.[1];
  const cs = trimmed.match(CITY_STATE_RE);
  const city = cs?.[1]?.trim();
  const state = cs?.[2]?.trim();

  return {
    raw: trimmed,
    city: city ?? fallbackCity,
    state: state ?? fallbackState,
    zip,
  };
}