import { DateTime } from 'luxon';
import type { ParsedAddress } from '../utils/address.js';

export type Lead = {
  source: string;
  url: string;
  title: string;
  price?: number;
  description?: string;
  addressText?: string;
  parsedAddress?: ParsedAddress;
  city?: string;
  state?: string;
  contact?: string;
  postedAtIso?: string;
};

export function normalize(input: Partial<Lead> & { source: string; url: string; title: string }): Lead {
  const price = sanitizePrice(input.price);
  return {
    source: input.source,
    url: input.url,
    title: input.title.trim(),
    price,
    description: input.description?.trim(),
    addressText: input.addressText?.trim(),
    parsedAddress: input.parsedAddress,
    city: input.city?.trim(),
    state: input.state?.trim(),
    contact: input.contact?.trim(),
    postedAtIso: input.postedAtIso ?? DateTime.now().toISO(),
  };
}

function sanitizePrice(p?: number | string): number | undefined {
  if (p === undefined) return undefined;
  if (typeof p === 'number') return p;
  const v = Number(String(p).replace(/[^0-9.]/g, ''));
  return Number.isFinite(v) ? v : undefined;
}