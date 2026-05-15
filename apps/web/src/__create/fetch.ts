export const fetchWithHeaders = async (
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> => {
  return fetch(input, init);
};

export default fetchWithHeaders;
