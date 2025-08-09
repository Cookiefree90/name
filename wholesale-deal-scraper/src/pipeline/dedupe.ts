import type { Lead } from './normalize.js';

export function dedupe(leads: Lead[]): Lead[] {
  const seen = new Set<string>();
  const out: Lead[] = [];
  for (const l of leads) {
    const key = `${l.source}|${l.url}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(l);
  }
  return out;
}