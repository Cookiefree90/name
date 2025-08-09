# Wholesale Deal Scraper (Node + Playwright)

Finds public wholesale leads from Craigslist real estate by-owner posts for a given market. Saves to CSV and optionally Google Sheets.

## Quick Start (Cursor)
1. **Create a new Node + TypeScript project** (or paste this repo into your workspace).
2. Copy `.env.example` → `.env` and set `MARKET`, `CITY`, `STATE`.
3. Install deps and browsers:
   ```bash
   npm i
   npx playwright install chromium
   ```
4. Run:
   ```bash
   npm run dev -- --market indianapolis --pages 2
   ```
5. Check `leads.csv`.

## Google Sheets (optional)
- Set `GOOGLE_SHEETS_ENABLED=true`, `GOOGLE_SHEETS_ID`, and `GOOGLE_SHEETS_TAB` in `.env`.
- Ensure your environment has Google ADC (Application Default Credentials): for local, `gcloud auth application-default login`.

## Extend
- Add sources in `src/sources/*`. Return `Lead[]` shaped by `normalize()`.
- Add phone/email extraction via regex or 3rd-party enrichment.
- Add geocoding (server-side rate limits!) to convert parsed addresses to lat/lng.

## Ethics & Compliance
- Obey robots.txt and ToS.
- Add delays and randomization.
- Provide an opt-out mechanism if you store personal info.