# 360 Estate Suite Web

This is the standalone web app exported from Anything and cleaned up to run without the Anything runtime.

## Stack

- React Router v7 with SSR
- Hono server adapter
- Supabase Postgres via `pg`
- Auth.js-compatible credentials auth
- Stripe Checkout and webhooks for credit purchases
- OpenAI APIs for AI staging and virtual-tour generation
- Supabase Storage for uploads, with local filesystem fallback under `storage/uploads`

## Setup

1. Install dependencies:

   ```bash
   npm install --legacy-peer-deps
   ```

2. Create `.env` from `.env.example` and fill the values in your deployment environment. Do not commit `.env`.

3. Create the database tables in Supabase:

   ```bash
   psql "$DATABASE_URL" -f migrations/001_init.sql
   ```

4. Create a public Supabase Storage bucket named `uploads`.

5. Start development:

   ```bash
   npm run dev
   ```

6. Build and run production:

   ```bash
   npm run build
   npm start
   ```

## Notes

- Uploaded files are stored in Supabase Storage when `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set. Local storage is only a development fallback.
- Set `APP_URL` and `AUTH_URL` to the public URL in production.
- Stripe webhooks should point to `/api/stripe/webhook`.
- `PDF_SERVICE_URL` is optional. If absent, the app uses the local fillable-PDF fallback.
