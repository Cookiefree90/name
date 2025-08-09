import 'dotenv/config';

export const CFG = {
  market: process.env.MARKET ?? 'indianapolis',
  city: process.env.CITY ?? 'indianapolis',
  state: process.env.STATE ?? 'IN',
  maxPages: Number(process.env.MAX_PAGES ?? '2'),
  outCsv: process.env.OUT_CSV ?? './leads.csv',
  google: {
    enabled: (process.env.GOOGLE_SHEETS_ENABLED ?? 'false').toLowerCase() === 'true',
    sheetId: process.env.GOOGLE_SHEETS_ID ?? '',
    tab: process.env.GOOGLE_SHEETS_TAB ?? 'Leads',
  },
};