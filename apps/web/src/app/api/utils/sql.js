import pg from "pg";

const { Pool } = pg;

const pool = process.env.DATABASE_URL
  ? new Pool({
      connectionString: process.env.DATABASE_URL,
      ssl:
        process.env.DATABASE_SSL === "false"
          ? false
          : { rejectUnauthorized: false },
    })
  : null;

function missingDatabaseUrl() {
  throw new Error(
    "No database connection string was provided. Set process.env.DATABASE_URL.",
  );
}

function buildQuery(stringsOrText, values = []) {
  if (Array.isArray(stringsOrText) && "raw" in stringsOrText) {
    let text = "";
    const params = [];

    stringsOrText.forEach((part, index) => {
      text += part;
      if (index < values.length) {
        params.push(values[index]);
        text += `$${params.length}`;
      }
    });

    return { text, values: params };
  }

  return {
    text: String(stringsOrText),
    values: Array.isArray(values) ? values : [],
  };
}

async function runQuery(client, stringsOrText, values) {
  const query = buildQuery(stringsOrText, values);
  const result = await client.query(query.text, query.values);
  return result.rows;
}

function createTxQuery() {
  return (stringsOrText, values) => buildQuery(stringsOrText, values);
}

async function sql(stringsOrText, values) {
  if (!pool) missingDatabaseUrl();
  return runQuery(pool, stringsOrText, values);
}

sql.transaction = async (callbackOrQueries) => {
  if (!pool) missingDatabaseUrl();

  const txQuery = createTxQuery();
  const queries =
    typeof callbackOrQueries === "function"
      ? callbackOrQueries(txQuery)
      : callbackOrQueries;

  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const results = [];

    for (const queryInput of queries) {
      if (!queryInput) {
        results.push([]);
        continue;
      }

      const result = await client.query(queryInput.text, queryInput.values);
      results.push(result.rows);
    }

    await client.query("COMMIT");
    return results;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
};

export default sql;
