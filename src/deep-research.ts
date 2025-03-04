import FirecrawlApp, { SearchResponse } from '@mendable/firecrawl-js';
import { generateObject } from 'ai';
import { compact } from 'lodash-es';
import pLimit from 'p-limit';
import { z } from 'zod';

import { o3MiniModel, trimPrompt } from './ai/providers';
import { systemPrompt } from './prompt';
import { OutputManager } from './output-manager';

// Initialize output manager for coordinated console/progress output
const output = new OutputManager();

// Replace console.log with output.log
function log(...args: any[]) {
  output.log(...args);
}

export type ResearchProgress = {
  currentDepth: number;
  totalDepth: number;
  currentBreadth: number;
  totalBreadth: number;
  currentQuery?: string;
  totalQueries: number;
  completedQueries: number;
};

type ResearchResult = {
  learnings: string[];
  visitedUrls: string[];
};

// increase this if you have higher API rate limits
const ConcurrencyLimit = 2;

// Rate limits for Firecrawl API - updated based on actual error message
const SCRAPE_RATE_LIMIT = 6; // 6 scrapes per minute (downgraded from 10 based on actual limit)
const CRAWL_RATE_LIMIT = 1;   // 1 crawl per minute

// Create rate limiters for each API type
class RateLimiter {
  private queue: number[] = [];
  private readonly rateLimit: number;
  private readonly timeWindow: number = 60000; // 1 minute in ms
  private isRateLimited: boolean = false;
  private rateLimitResetTime: number = 0;

  constructor(rateLimit: number) {
    this.rateLimit = rateLimit;
  }

  async waitForSlot(): Promise<void> {
    const now = Date.now();
    
    // If we're in a rate limited state, wait until reset time
    if (this.isRateLimited && now < this.rateLimitResetTime) {
      const waitTime = this.rateLimitResetTime - now;
      log(`Rate limit active, waiting ${waitTime}ms until reset`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      this.isRateLimited = false;
      this.queue = []; // Clear queue after waiting
      return;
    }
    
    // Remove timestamps older than time window
    this.queue = this.queue.filter(timestamp => timestamp > now - this.timeWindow);
    
    // If we're at or over rate limit, wait until the oldest request expires
    if (this.queue.length >= this.rateLimit && this.queue.length > 0) {
      // We've verified queue has at least one element, so oldestTimestamp can't be undefined
      const oldestTimestamp = this.queue[0];
      // But TypeScript might not recognize this, so add a fallback just in case
      const waitTime = (oldestTimestamp ?? now) + this.timeWindow - now + 1000; // Add 1s buffer
      
      if (waitTime > 0) {
        log(`Rate limit reached, waiting ${waitTime}ms`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
        // Recursively check again after waiting
        return this.waitForSlot();
      }
    }
    
    // Add current timestamp to queue
    this.queue.push(now);
  }

  // Call this when we receive a 429 response
  setRateLimited(resetTimeStr?: string) {
    this.isRateLimited = true;
    
    // If reset time was provided, use it
    if (resetTimeStr) {
      try {
        this.rateLimitResetTime = new Date(resetTimeStr).getTime();
      } catch (e) {
        // If parsing fails, use default wait time of 10 seconds
        this.rateLimitResetTime = Date.now() + 10000;
      }
    } else {
      // Default: wait 10 seconds
      this.rateLimitResetTime = Date.now() + 10000;
    }
  }
}

const scrapeRateLimiter = new RateLimiter(SCRAPE_RATE_LIMIT);
const crawlRateLimiter = new RateLimiter(CRAWL_RATE_LIMIT);

// Initialize Firecrawl with optional API key and optional base url

const firecrawl = new FirecrawlApp({
  apiKey: process.env.FIRECRAWL_KEY ?? '',
  apiUrl: process.env.FIRECRAWL_BASE_URL,
});

// take en user query, return a list of SERP queries
async function generateSerpQueries({
  query,
  numQueries = 3,
  learnings,
}: {
  query: string;
  numQueries?: number;

  // optional, if provided, the research will continue from the last learning
  learnings?: string[];
}) {
  const res = await generateObject({
    model: o3MiniModel,
    system: systemPrompt(),
    prompt: `Given the following prompt from the user, generate a list of SERP queries to research the topic. Return a maximum of ${numQueries} queries, but feel free to return less if the original prompt is clear. Make sure each query is unique and not similar to each other: <prompt>${query}</prompt>\n\n${
      learnings
        ? `Here are some learnings from previous research, use them to generate more specific queries: ${learnings.join(
            '\n',
          )}`
        : ''
    }`,
    schema: z.object({
      queries: z
        .array(
          z.object({
            query: z.string().describe('The SERP query'),
            researchGoal: z
              .string()
              .describe(
                'First talk about the goal of the research that this query is meant to accomplish, then go deeper into how to advance the research once the results are found, mention additional research directions. Be as specific as possible, especially for additional research directions.',
              ),
          }),
        )
        .describe(`List of SERP queries, max of ${numQueries}`),
    }),
  });
  log(
    `Created ${res.object.queries.length} queries`,
    res.object.queries,
  );

  return res.object.queries.slice(0, numQueries);
}

async function processSerpResult({
  query,
  result,
  numLearnings = 3,
  numFollowUpQuestions = 3,
}: {
  query: string;
  result: SearchResponse;
  numLearnings?: number;
  numFollowUpQuestions?: number;
}) {
  const contents = compact(result.data.map(item => item.markdown)).map(
    content => trimPrompt(content, 25_000),
  );
  log(`Ran ${query}, found ${contents.length} contents`);

  const res = await generateObject({
    model: o3MiniModel,
    abortSignal: AbortSignal.timeout(60_000),
    system: systemPrompt(),
    prompt: `Given the following contents from a SERP search for the query <query>${query}</query>, generate a list of learnings from the contents. Return a maximum of ${numLearnings} learnings, but feel free to return less if the contents are clear. Make sure each learning is unique and not similar to each other. The learnings should be concise and to the point, as detailed and information dense as possible. Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.\n\n<contents>${contents
      .map(content => `<content>\n${content}\n</content>`)
      .join('\n')}</contents>`,
    schema: z.object({
      learnings: z
        .array(z.string())
        .describe(`List of learnings, max of ${numLearnings}`),
      followUpQuestions: z
        .array(z.string())
        .describe(
          `List of follow-up questions to research the topic further, max of ${numFollowUpQuestions}`,
        ),
    }),
  });
  log(
    `Created ${res.object.learnings.length} learnings`,
    res.object.learnings,
  );

  return res.object;
}

export async function writeFinalReport({
  prompt,
  learnings,
  visitedUrls,
}: {
  prompt: string;
  learnings: string[];
  visitedUrls: string[];
}) {
  const learningsString = trimPrompt(
    learnings
      .map(learning => `<learning>\n${learning}\n</learning>`)
      .join('\n'),
    150_000,
  );

  const res = await generateObject({
    model: o3MiniModel,
    system: systemPrompt(),
    prompt: `Given the following prompt from the user, write a final report on the topic using the learnings from research. Make it as as detailed as possible, aim for 3 or more pages, include ALL the learnings from research:\n\n<prompt>${prompt}</prompt>\n\nHere are all the learnings from previous research:\n\n<learnings>\n${learningsString}\n</learnings>`,
    schema: z.object({
      reportMarkdown: z
        .string()
        .describe('Final report on the topic in Markdown'),
    }),
  });

  // Append the visited URLs section to the report
  const urlsSection = `\n\n## Sources\n\n${visitedUrls.map(url => `- ${url}`).join('\n')}`;
  return res.object.reportMarkdown + urlsSection;
}

// Rate-limited version of firecrawl.search with retry logic
async function rateLimitedSearch(query: string, options: any, retryCount = 0): Promise<any> {
  const MAX_RETRIES = 3;
  const RETRY_DELAYS = [1000, 5000, 15000]; // Exponential backoff delays
  
  try {
    await scrapeRateLimiter.waitForSlot();
    return await firecrawl.search(query, options);
  } catch (error: any) {
    // Check if it's a rate limit error (429)
    if (error?.message?.includes('429') && error?.message?.includes('Rate limit exceeded')) {
      // Parse reset time if available in the error message
      const resetTimeMatch = error.message.match(/resets at (.*?)(?:\s|$)/);
      const resetTime = resetTimeMatch ? resetTimeMatch[1] : undefined;
      
      // Inform the rate limiter about the 429 error
      scrapeRateLimiter.setRateLimited(resetTime);
      
      // Retry if under max retries
      if (retryCount < MAX_RETRIES) {
        const delay = RETRY_DELAYS[retryCount] || 15000;
        log(`Rate limited (429), retrying in ${delay}ms (attempt ${retryCount + 1}/${MAX_RETRIES})`);
        await new Promise(resolve => setTimeout(resolve, delay));
        return rateLimitedSearch(query, options, retryCount + 1);
      }
    }
    
    // Re-throw the error if we can't handle it or max retries exceeded
    throw error;
  }
}

// Modify the deepResearch function to use a lower concurrency
export async function deepResearch({
  query,
  breadth,
  depth,
  learnings = [],
  visitedUrls = [],
  onProgress,
}: {
  query: string;
  breadth: number;
  depth: number;
  learnings?: string[];
  visitedUrls?: string[];
  onProgress?: (progress: ResearchProgress) => void;
}): Promise<ResearchResult> {
  // Use a more conservative concurrency limit to avoid rate limiting
  const actualConcurrencyLimit = Math.min(ConcurrencyLimit, 1); // Use 1 for strict sequential processing
  const limit = pLimit(actualConcurrencyLimit);

  const progress: ResearchProgress = {
    currentDepth: depth,
    totalDepth: depth,
    currentBreadth: breadth,
    totalBreadth: breadth,
    totalQueries: 0,
    completedQueries: 0,
  };
  
  const reportProgress = (update: Partial<ResearchProgress>) => {
    Object.assign(progress, update);
    onProgress?.(progress);
  };

  const serpQueries = await generateSerpQueries({
    query,
    learnings,
    numQueries: breadth,
  });
  
  reportProgress({
    totalQueries: serpQueries.length,
    currentQuery: serpQueries[0]?.query
  });
  
  const results = await Promise.all(
    serpQueries.map(serpQuery =>
      limit(async () => {
        try {
          const result = await rateLimitedSearch(serpQuery.query, {
            timeout: 15000,
            limit: 5,
            scrapeOptions: { formats: ['markdown'] },
          });

          // Collect URLs from this search with proper type safety
          const newUrls: string[] = [];
          for (const item of result.data) {
            if (item && typeof item.url === 'string') {
              newUrls.push(item.url);
            }
          }
          const newBreadth = Math.ceil(breadth / 2);
          const newDepth = depth - 1;

          const newLearnings = await processSerpResult({
            query: serpQuery.query,
            result,
            numFollowUpQuestions: newBreadth,
          });
          const allLearnings = [...learnings, ...newLearnings.learnings];
          const allUrls = [...visitedUrls, ...newUrls];

          if (newDepth > 0) {
            log(
              `Researching deeper, breadth: ${newBreadth}, depth: ${newDepth}`,
            );

            reportProgress({
              currentDepth: newDepth,
              currentBreadth: newBreadth,
              completedQueries: progress.completedQueries + 1,
              currentQuery: serpQuery.query,
            });

            const nextQuery = `
            Previous research goal: ${serpQuery.researchGoal}
            Follow-up research directions: ${newLearnings.followUpQuestions.map(q => `\n${q}`).join('')}
          `.trim();

            return deepResearch({
              query: nextQuery,
              breadth: newBreadth,
              depth: newDepth,
              learnings: allLearnings,
              visitedUrls: allUrls,
              onProgress,
            });
          } else {
            reportProgress({
              currentDepth: 0,
              completedQueries: progress.completedQueries + 1,
              currentQuery: serpQuery.query,
            });
            return {
              learnings: allLearnings,
              visitedUrls: allUrls,
            };
          }
        } catch (e: any) {
          if (e.message && e.message.includes('Timeout')) {
            log(
              `Timeout error running query: ${serpQuery.query}: `,
              e,
            );
          } else {
            log(`Error running query: ${serpQuery.query}: `, e);
          }
          return {
            learnings: [],
            visitedUrls: [],
          };
        }
      }),
    ),
  );

  // We need to filter for valid results and ensure type safety
  const safeResults = results.filter(Boolean) as Array<{ 
    learnings: string[]; 
    visitedUrls: string[]; 
  }>;
  
  // Safely extract all learnings as strings
  const allLearnings: string[] = [];
  for (const result of safeResults) {
    if (Array.isArray(result.learnings)) {
      for (const learning of result.learnings) {
        if (typeof learning === 'string') {
          allLearnings.push(learning);
        }
      }
    }
  }
  
  // Safely extract all URLs as strings
  const allVisitedUrls: string[] = [];
  for (const result of safeResults) {
    if (Array.isArray(result.visitedUrls)) {
      for (const url of result.visitedUrls) {
        if (typeof url === 'string') {
          allVisitedUrls.push(url);
        }
      }
    }
  }
  
  return {
    learnings: [...new Set(allLearnings)],
    visitedUrls: [...new Set(allVisitedUrls)],
  };
}
