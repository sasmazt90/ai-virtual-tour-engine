const nativeFetch = globalThis.fetch?.bind(globalThis);

export const fetchWithHeaders = async (
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> => {
  if (!nativeFetch) {
    throw new Error('fetch is not available in this environment');
  }
  return nativeFetch(input, init);
};

export default fetchWithHeaders;
