export interface TokenizedChunk {
  id: string;
  text: string;
  tokens: string[];
  start: number;
  end: number;
}

export function tokenizeAndChunk(text: string, chunkSize: number = 512, overlap: number = 50): TokenizedChunk[] {
  const tokens = text.split(/\s+/);
  const chunks: TokenizedChunk[] = [];
  let start = 0;

  while (start < tokens.length) {
    const end = Math.min(start + chunkSize, tokens.length);
    const chunkTokens = tokens.slice(start, end);
    const chunkText = chunkTokens.join(' ');

    chunks.push({
      id: `chunk_${chunks.length}`,
      text: chunkText,
      tokens: chunkTokens,
      start,
      end,
    });

    start = end - overlap;
    if (start <= 0) break; // prevent infinite loop
  }

  return chunks;
}