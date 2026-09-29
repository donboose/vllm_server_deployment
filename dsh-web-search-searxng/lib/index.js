import z from "@deepseek-ai/schemastery";
import { WebError } from "@deepseek-ai/dsh-web";
//#region lib/types/provider.js
/**
 * Self-hosted SearXNG search provider for the web capability seam (`ctx.web`).
 * Runs the `web_search` tool against a free, keyless SearXNG JSON API (the
 * user's own instance on the shared GPU box) so search never goes through a
 * billed LLM API. When the instance is unreachable or yields nothing, an
 * optional DuckDuckGo HTML scrape covers the gap.
 * @module dsh-web-search-searxng/provider
 */
/** Cordis plugin name used by loader diagnostics. */
const name = "web-search-searxng";
/** The web seam this provider registers into. */
const inject = ["web"];
/** Stable id this provider registers under. */
const SEARXNG_PROVIDER_ID = "searxng";
/**
 * Default SearXNG instance base URL — the user's own self-hosted instance on
 * the shared GPU box (same host as the local model). Override per profile.
 */
const DEFAULT_SEARXNG_URL = "http://10.1.45.74:8888";
/**
 * Fallback anonymous search: DuckDuckGo's HTML endpoint. Unofficial layout;
 * used only when the SearXNG instance is unreachable or yields nothing.
 */
const DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/";
/** User agent sent to both backends. */
const USER_AGENT = "deepseek-harness/0.2.0 (+dsh-web-search-searxng)";
/** Settings section for this provider. */
const Config = z.object({
	/** Base URL of the SearXNG instance (no trailing path). */
	searxngUrl: z.string().default(DEFAULT_SEARXNG_URL),
	/** Fall back to DuckDuckGo HTML when SearXNG fails or yields no results. */
	useDuckDuckGoFallback: z.boolean().default(true)
});
/** Decode the handful of HTML entities DuckDuckGo's markup uses. */
function decodeEntities(text) {
	return text
		.replace(/&amp;/g, "&")
		.replace(/&quot;/g, '"')
		.replace(/&#x27;/g, "'")
		.replace(/&#39;/g, "'")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">");
}
/** Strip inline tags and collapse whitespace. */
function cleanText(text) {
	return decodeEntities(text.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();
}
/**
 * Match an `<a>` tag carrying the given class, tolerating attribute order.
 * @param block - one DuckDuckGo result block.
 * @param className - the anchor class to match (`result__a`, `result__snippet`).
 * @returns the regex match (href = group 1, inner HTML = group 2), or undefined.
 */
function matchAnchor(block, className) {
	const classFirst = block.match(new RegExp(`<a[^>]*class="${className}"[^>]*href="([^"]+)"[^>]*>([\\s\\S]*?)</a>`));
	if (classFirst !== null) return classFirst;
	return block.match(new RegExp(`<a[^>]*href="([^"]+)"[^>]*class="${className}"[^>]*>([\\s\\S]*?)</a>`));
}
/**
 * Query one SearXNG instance's JSON API and normalize its results.
 * @param baseUrl - the instance base URL (no trailing path).
 * @param request - the seam's search request.
 * @param signal - optional cancellation.
 * @returns the normalized result (the seam enforces `maxResults`).
 * @throws {WebError} `WEB_PROVIDER_ERROR` on unreachable/HTTP/parse failure or zero results.
 */
async function searchSearxng(baseUrl, request, signal) {
	const url = new URL("/search", baseUrl);
	url.searchParams.set("q", request.query);
	url.searchParams.set("format", "json");
	let response;
	try {
		response = await fetch(url, {
			method: "GET",
			redirect: "error",
			headers: {
				accept: "application/json",
				"user-agent": USER_AGENT
			},
			...signal !== void 0 ? { signal } : {}
		});
	} catch (error) {
		throw new WebError(`SearXNG search request to ${url.origin} failed: ${String(error)}`, "WEB_PROVIDER_ERROR", { cause: error });
	}
	if (!response.ok) throw new WebError(`SearXNG HTTP error (HTTP ${response.status}) from ${url.origin}`, "WEB_PROVIDER_ERROR");
	let data;
	try {
		data = await response.json();
	} catch (error) {
		throw new WebError(`SearXNG returned an unprocessable response body: ${String(error)}`, "WEB_PROVIDER_ERROR", { cause: error });
	}
	const results = Array.isArray(data?.results) ? data.results : [];
	const sources = [];
	for (const item of results) {
		if (typeof item?.url !== "string" || item.url.length === 0) continue;
		sources.push({
			url: item.url,
			...typeof item.title === "string" && item.title.length > 0 ? { title: item.title } : {},
			...typeof item.content === "string" && item.content.length > 0 ? { snippet: item.content } : {},
			...typeof item.publishedDate === "string" && item.publishedDate.length > 0 ? { publishedAt: item.publishedDate } : {}
		});
	}
	if (sources.length === 0) throw new WebError("SearXNG returned no results", "WEB_PROVIDER_ERROR");
	return {
		sources,
		truncated: false
	};
}
/**
 * Scrape DuckDuckGo's anonymous HTML endpoint as a last resort.
 * @param request - the seam's search request.
 * @param signal - optional cancellation.
 * @returns the normalized result (the seam enforces `maxResults`).
 * @throws {WebError} `WEB_PROVIDER_ERROR` on failure or when nothing parseable comes back.
 */
async function searchDuckDuckGo(request, signal) {
	const url = new URL(DUCKDUCKGO_HTML_URL);
	url.searchParams.set("q", request.query);
	let response;
	try {
		response = await fetch(url, {
			method: "GET",
			redirect: "error",
			headers: {
				accept: "text/html",
				"user-agent": USER_AGENT
			},
			...signal !== void 0 ? { signal } : {}
		});
	} catch (error) {
		throw new WebError(`DuckDuckGo search request failed: ${String(error)}`, "WEB_PROVIDER_ERROR", { cause: error });
	}
	if (!response.ok) throw new WebError(`DuckDuckGo HTTP error (HTTP ${response.status})`, "WEB_PROVIDER_ERROR");
	let html;
	try {
		html = await response.text();
	} catch (error) {
		throw new WebError(`DuckDuckGo response could not be read: ${String(error)}`, "WEB_PROVIDER_ERROR", { cause: error });
	}
	const sources = [];
	for (const block of html.split(/class="result results_links /).slice(1)) {
		const anchor = matchAnchor(block, "result__a");
		if (anchor === void 0) continue;
		let link = anchor[1];
		const uddg = link.match(/[?&]uddg=([^&]+)/);
		if (uddg !== null) link = decodeURIComponent(uddg[1]);
		else if (link.startsWith("//")) link = `https:${link}`;
		if (!/^https?:\/\//.test(link)) continue;
		const title = cleanText(anchor[2]);
		const snippet = matchAnchor(block, "result__snippet");
		sources.push({
			url: link,
			...title.length > 0 ? { title } : {},
			...snippet !== void 0 ? { snippet: cleanText(snippet[2]) } : {}
		});
	}
	if (sources.length === 0) throw new WebError("DuckDuckGo returned no parseable results (page layout may have changed)", "WEB_PROVIDER_ERROR");
	return {
		sources,
		truncated: false
	};
}
/** Register the SearXNG search provider (with DuckDuckGo fallback) with `ctx.web`. */
function apply(ctx, config) {
	const { searxngUrl, useDuckDuckGoFallback } = config;
	ctx.web.registerSearchProvider({
		id: SEARXNG_PROVIDER_ID,
		/** No credentials to check: a reachable instance (or the fallback) makes it usable. */
		available: () => URL.canParse(searxngUrl) || useDuckDuckGoFallback,
		async search(request, signal) {
			const errors = [];
			try {
				return await searchSearxng(searxngUrl, request, signal);
			} catch (error) {
				// An already-cancelled search is not a reason to hit the fallback.
				if (signal?.aborted === true || !useDuckDuckGoFallback) throw error;
				errors.push(error);
			}
			try {
				return await searchDuckDuckGo(request, signal);
			} catch (error) {
				errors.push(error);
			}
			throw new WebError(`SearXNG web search failed: ${errors.map((error) => error.message).join("; ")}`, "WEB_PROVIDER_ERROR", {
				cause: errors[0]
			});
		}
	});
}
//#endregion
export { Config, DUCKDUCKGO_HTML_URL, DEFAULT_SEARXNG_URL, SEARXNG_PROVIDER_ID, apply, inject, name };
