import { createObjectCsvWriter } from 'csv-writer';
import type { Lead } from './normalize.js';
import { CFG } from '../config.js';
import { log } from '../utils/log.js';
import { googleSheetsAppend } from './sheets.js';

export async function exportLeads(leads: Lead[]) {
  if (!leads.length) return log.warn('No leads to export.');

  const csv = createObjectCsvWriter({
    path: CFG.outCsv,
    header: [
      { id: 'source', title: 'source' },
      { id: 'url', title: 'url' },
      { id: 'title', title: 'title' },
      { id: 'price', title: 'price' },
      { id: 'description', title: 'description' },
      { id: 'addressText', title: 'addressText' },
      { id: 'city', title: 'city' },
      { id: 'state', title: 'state' },
      { id: 'contact', title: 'contact' },
      { id: 'postedAtIso', title: 'postedAtIso' }
    ],
    alwaysQuote: true,
  });

  await csv.writeRecords(leads);
  log.info(`Wrote ${leads.length} leads to`, CFG.outCsv);

  if (CFG.google.enabled) {
    await googleSheetsAppend(leads);
  }
}