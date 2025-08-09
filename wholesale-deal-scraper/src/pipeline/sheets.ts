import { google } from 'googleapis';
import path from 'node:path';
import process from 'node:process';
import type { Lead } from './normalize.js';
import { CFG } from '../config.js';
import { log } from '../utils/log.js';

export async function googleSheetsAppend(leads: Lead[]) {
  if (!CFG.google.sheetId) return log.warn('GOOGLE_SHEETS_ID not set; skipping Sheets export.');
  const auth = new google.auth.GoogleAuth({
    scopes: ['https://www.googleapis.com/auth/spreadsheets']
  });

  const sheets = google.sheets({ version: 'v4', auth });
  const values = leads.map(l => [
    l.source, l.url, l.title, l.price ?? '', l.description ?? '', l.addressText ?? '', l.city ?? '', l.state ?? '', l.contact ?? '', l.postedAtIso ?? ''
  ]);

  await sheets.spreadsheets.values.append({
    spreadsheetId: CFG.google.sheetId,
    range: `${CFG.google.tab}!A1`,
    valueInputOption: 'RAW',
    requestBody: { values }
  });

  log.info(`Appended ${leads.length} rows to Google Sheets tab "${CFG.google.tab}".`);
}