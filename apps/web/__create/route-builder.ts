import { Hono } from 'hono';
import type { Handler } from 'hono/types';
import updatedFetch from '../src/__create/fetch';

const API_BASENAME = '/api';
const api = new Hono();

if (globalThis.fetch) {
  globalThis.fetch = updatedFetch;
}

type RouteModule = Record<string, unknown>;

const routeModules = import.meta.glob<RouteModule>('../src/app/api/**/route.js', {
  eager: true,
});

function getHonoPath(routeFile: string): string {
  const relativePath = routeFile
    .replace('../src/app/api', '')
    .replace(/\/route\.js$/, '');

  if (!relativePath) return '/';

  const parts = relativePath.split('/').filter(Boolean);
  const transformedParts = parts.map((segment) => {
    const match = segment.match(/^\[(\.{3})?([^\]]+)\]$/);
    if (!match) return segment;

    const [, dots, param] = match;
    return dots === '...' ? `:${param}{.+}` : `:${param}`;
  });

  return `/${transformedParts.join('/')}`;
}

function getParams(pattern: string, c: Parameters<Handler>[0]) {
  return pattern === '/' ? {} : c.req.param();
}

for (const [routeFile, route] of Object.entries(routeModules).sort(
  ([a], [b]) => b.length - a.length
)) {
  const honoPath = getHonoPath(routeFile);

  for (const method of ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'] as const) {
    const routeHandler = route[method];
    if (typeof routeHandler !== 'function') continue;

    const handler: Handler = async (c) => {
      return await routeHandler(c.req.raw, { params: getParams(honoPath, c) });
    };

    switch (method) {
      case 'GET':
        api.get(honoPath, handler);
        break;
      case 'POST':
        api.post(honoPath, handler);
        break;
      case 'PUT':
        api.put(honoPath, handler);
        break;
      case 'DELETE':
        api.delete(honoPath, handler);
        break;
      case 'PATCH':
        api.patch(honoPath, handler);
        break;
    }
  }
}

export { api, API_BASENAME };
