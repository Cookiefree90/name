import { chromium, type Browser } from 'playwright';
import { sleep } from '../utils/sleep.js';
import { log } from '../utils/log.js';
import { parseAddress } from '../utils/address.js';
import { normalize, type Lead } from '../pipeline/normalize.js';
import { CFG } from '../config.js';

export async function scrapeCraigslist(): Promise<Lead[]> {
  const browser: Browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ userAgent: 'Mozilla/5.0 (compatible; DealScraper/0.1)' });
  const page = await context.newPage();

  const base = `https://${CFG.market}.craigslist.org/search/rea?sort=date&availabilityMode=0&sale_date=all+dates`;
  const leads: Lead[] = [];

  for (let p = 0; p < CFG.maxPages; p++) {
    const url = `${base}&s=${p * 120}`; // 120 per page
    log.info('Craigslist page:', url);
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.waitForSelector('.result-row, .cl-results-page');

    const links = await page.$$eval('a.result-title', as => as.map(a => (a as HTMLAnchorElement).href));

    for (const href of links) {
      try {
        await page.goto(href, { waitUntil: 'domcontentloaded', timeout: 45000 });
        await page.waitForSelector('#titletextonly, .postingtitletext', { timeout: 10000 });

        const title = await page.$eval('#titletextonly, .postingtitletext', el => el.textContent?.trim() ?? '');
        const priceText = await page.$('.price') ? await page.$eval('.price', el => el.textContent ?? '') : '';
        const price = priceText ? Number(priceText.replace(/[^0-9.]/g, '')) : undefined;
        const description = await page.$('#postingbody') ? await page.$eval('#postingbody', el => el.textContent ?? '') : undefined;
        const mapAddress = await page.$('.mapaddress') ? await page.$eval('.mapaddress', el => el.textContent ?? '') : undefined;
        const parsedAddress = parseAddress(mapAddress ?? undefined, CFG.city, CFG.state);
        const contact = await page.$('p.reply-tel-number') ? await page.$eval('p.reply-tel-number', el => el.textContent ?? '') : undefined;

        leads.push(normalize({
          source: 'craigslist',
          url: href,
          title,
          price,
          description,
          addressText: mapAddress,
          parsedAddress,
          city: parsedAddress?.city ?? CFG.city,
          state: parsedAddress?.state ?? CFG.state,
          contact,
        }));

        await sleep(500 + Math.random() * 500);
      } catch (e) {
        log.warn('Craigslist item error:', (e as Error).message);
      }
    }

    await sleep(1000 + Math.random() * 1000);
  }

  await browser.close();
  return leads;
}