# 360 Estate Suite

This repository now contains the standalone 360 Estate Suite web app exported from Anything and cleaned up to run without the Anything platform runtime.

## Apps

- `apps/web` - the production web application: React Router SSR, Hono API server, Supabase Postgres, Supabase Storage, Auth.js-style credentials auth, Stripe, and OpenAI.
- Root Python files and `services/` - the earlier virtual-tour engine prototype kept for reference.

## Run the Web App

```bash
cd apps/web
npm install --legacy-peer-deps
```

Create `apps/web/.env` from `apps/web/.env.example`, then initialize Supabase Postgres:

```bash
psql "$DATABASE_URL" -f migrations/001_init.sql
```

Create a public Supabase Storage bucket named `uploads`, then set:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_STORAGE_BUCKET=uploads`

Start development:

```bash
npm run dev
```

Build and run production:

```bash
npm run build
npm start
```

## Required Environment

- `DATABASE_URL`
- `DATABASE_SSL`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_STORAGE_BUCKET`
- `AUTH_SECRET`
- `APP_URL`
- `AUTH_URL`
- `OPEN_AI_API_KEY`
- `STRIPE_SECRET_KEY`
- `STRIPE_PUBLISHABLE_KEY`
- `STRIPE_WEBHOOK_KEY`

Optional:

- `GOOGLE_MAPS_API_KEY`
- `PDF_SERVICE_URL`

Never commit `.env` files or live API keys.
