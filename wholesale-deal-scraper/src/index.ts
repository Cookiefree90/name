import { Command } from 'commander';
import { log } from './utils/log.js';
import { exportLeads } from './pipeline/export.js';
import { dedupe } from './pipeline/dedupe.js';
import { scrapeCraigslist } from './sources/craigslist.js';
import { CFG } from './config.js';

const program = new Command();
program
  .name('wholesale-deal-scraper')
  .description('Scrape public wholesale leads and export to CSV/Sheets')
  .option('-m, --market <slug>', 'Craigslist market subdomain (e.g., indianapolis)', CFG.market)
  .option('-p, --pages <n>', 'Pages to scan (120 items per page)', String(CFG.maxPages))
  .parse(process.argv);

(async () => {
  const opts = program.opts();
  if (opts.market) process.env.MARKET = opts.market;
  if (opts.pages) process.env.MAX_PAGES = String(opts.pages);

  log.info('Starting scrape for market:', process.env.MARKET);

  const batches = await Promise.all([
    scrapeCraigslist(),
    // scrapeZillowFsbo(), // disabled
  ]);

  const leads = dedupe(batches.flat());
  await exportLeads(leads);

  log.info('Done.');
})();