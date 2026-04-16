import { chromium } from "playwright";
import fs from "fs-extra";
import slugify from "slugify";
import { DataChunking } from "./rag";


// ==========================
// CONFIG
// ==========================
const MAX_PAGES = 50;
const MAX_DEPTH = 3;
const DELAY_MS = 300;

// ==========================
// MAIN FUNCTION
// ==========================
export async function crawlDocs(START_URL) {
  const DOMAIN = new URL(START_URL).origin;

  const visited = new Set();
  const queue = [{ url: START_URL, depth: 0 }];

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  let count = 0;

  while (queue.length > 0 && count < MAX_PAGES) {
    const { url, depth } = queue.shift();

    if (!url || visited.has(url) || depth > MAX_DEPTH) continue;
    visited.add(url);

    console.log(`🌐 Visiting: ${url}`);

    try {
      await page.goto(url, { waitUntil: "networkidle" });

      // ==========================
      // 1. EXTRACT CLEAN CONTENT
      // ==========================
      const content = await extractContent(page);

      if (content && content.length > 200) {
        await savePage(url, content);

        // ✅ PASS DIRECTLY TO YOUR CHUNKER
        await DataChunking({
          url,
          content,
          source: DOMAIN
        });
      }

      // ==========================
      // 2. EXTRACT LINKS (NAV FIRST, FALLBACK TO ALL LINKS)
      // ==========================
      const links = await extractLinks(page);

      for (const link of links) {
        if (isValidLink(link, DOMAIN, visited)) {
          queue.push({ url: link, depth: depth + 1 });
        }
      }

      count++;
      await sleep(DELAY_MS);

    } catch (err) {
      console.error("❌ Error:", url, err.message);
    }
  }

  await browser.close();
  console.log("✅ Crawl finished");
}

// ==========================
// HELPERS
// ==========================

// ✅ Clean + structured extraction
async function extractContent(page) {
  return await page.evaluate(() => {
    // Remove junk
    ["nav", "footer", "aside", "script", "style"].forEach(tag => {
      document.querySelectorAll(tag).forEach(el => el.remove());
    });

    const root =
      document.querySelector("main") ||
      document.querySelector("article") ||
      document.body;

    const parts = [];

    root.querySelectorAll("h1, h2, h3, p, li, code").forEach(el => {
      const text = el.innerText?.trim();
      if (text) parts.push(text);
    });

    return parts.join("\n");
  });
}

// 🔥 NAV FIRST (BEST PRACTICE) + FALLBACK
async function extractLinks(page) {
  return await page.evaluate(() => {
    let elements = document.querySelectorAll("nav a");

    // fallback if no nav (important for some docs sites)
    if (!elements || elements.length === 0) {
      elements = document.querySelectorAll("a");
    }

    return Array.from(elements)
      .map(a => a.href)
      .filter(Boolean);
  });
}

// ✅ Strict filtering
function isValidLink(link, DOMAIN, visited) {
  return (
    link.startsWith(DOMAIN) &&
    !link.includes("#") &&
    !link.includes("?") &&
    !link.match(/\.(png|jpg|jpeg|gif|svg|pdf)$/) &&
    !visited.has(link)
  );
}

// ✅ Save cleaned page
async function savePage(url, content) {
  const slug = slugify(url, { strict: true }) || "home";

  await fs.ensureDir("./data");

  await fs.writeJson(`./data/${slug}.json`, {
    url,
    content
  }, { spaces: 2 });

  console.log("💾 Saved:", slug);
}

// ✅ Rate limit
const sleep = (ms) => new Promise(r => setTimeout(r, ms));