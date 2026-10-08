// Remote MCP server as an AWS Lambda (Function URL, payload v2).
//
// Transport: Streamable HTTP, stateless, JSON responses only (no SSE, no
// sessions), via the SDK's web-standard transport. A stateless transport serves
// one request, so each invocation connects the server to a fresh one (a Lambda
// instance handles one invocation at a time). The tools are the same ones the
// stdio and HTTP servers in server.ts register.
//
// The request path is ignored, so the function can sit behind any path prefix
// (e.g. /youtube/mcp).
//
// Handler: dist/lambda.handler

import { WebStandardStreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js';
import { createMcpServer } from './server.js';

interface FunctionUrlEvent {
  rawPath?: string;
  rawQueryString?: string;
  headers?: Record<string, string>;
  body?: string;
  isBase64Encoded?: boolean;
  requestContext?: { http?: { method?: string } };
}

interface FunctionUrlResult {
  statusCode: number;
  headers: Record<string, string>;
  body: string;
}

const CORS: Record<string, string> = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Accept, Authorization, Mcp-Session-Id, Mcp-Protocol-Version',
  'Access-Control-Expose-Headers': 'Mcp-Session-Id, Mcp-Protocol-Version',
};

function reply(statusCode: number, body = '', contentType?: string): FunctionUrlResult {
  return { statusCode, headers: { ...CORS, ...(contentType && { 'Content-Type': contentType }) }, body };
}

const server = createMcpServer();  // reused across warm invocations

export async function handler(event: FunctionUrlEvent): Promise<FunctionUrlResult> {
  const method = event.requestContext?.http?.method ?? 'POST';
  const headers = new Headers(event.headers ?? {});

  if (method === 'OPTIONS') return reply(204);
  if (method === 'GET') {
    // No server-initiated SSE stream in stateless mode; a plain browser visit gets a pointer.
    if ((headers.get('accept') ?? '').includes('text/event-stream')) return reply(405);
    return reply(200, 'YouTube MCP server (Streamable HTTP). POST JSON-RPC here.\n', 'text/plain; charset=utf-8');
  }
  if (method !== 'POST') return reply(405);

  // The SDK insists clients accept both, but responses here are always JSON; don't
  // turn away ones (e.g. curl's */*) that are less specific.
  headers.set('accept', 'application/json, text/event-stream');
  if (!headers.get('content-type')) headers.set('content-type', 'application/json');
  const body = event.isBase64Encoded ? Buffer.from(event.body ?? '', 'base64') : event.body;
  const url = `https://lambda${event.rawPath ?? '/'}${event.rawQueryString ? `?${event.rawQueryString}` : ''}`;

  const transport = new WebStandardStreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
    enableJsonResponse: true,
  });
  await server.connect(transport);
  try {
    const response = await transport.handleRequest(new Request(url, { method, headers, body }));
    return {
      statusCode: response.status,
      headers: { ...CORS, ...Object.fromEntries(response.headers) },
      body: await response.text(),
    };
  } finally {
    await server.close();
  }
}
