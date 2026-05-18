ALTER TABLE credit_transactions
  ADD COLUMN IF NOT EXISTS provider text,
  ADD COLUMN IF NOT EXISTS provider_ref text;

CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_transactions_provider_ref
  ON credit_transactions(provider, provider_ref)
  WHERE provider IS NOT NULL AND provider_ref IS NOT NULL;
