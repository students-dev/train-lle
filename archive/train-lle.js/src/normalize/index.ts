import { normalize } from 'unorm';
import { ExtractedArtifact } from '../extract/index.js';

export interface NormalizedArtifact {
  path: string;
  text: string;
  metadata: Record<string, any>;
}

export function normalizeText(artifact: ExtractedArtifact): NormalizedArtifact {
  let text = normalize(artifact.text, 'NFC');

  // Simple boilerplate removal: remove lines with many non-alphanumeric chars or short lines
  const lines = text.split('\n');
  const filtered = lines.filter((line: string) => {
    const trimmed = line.trim();
    if (trimmed.length < 10) return false;
    const alphaNum = trimmed.replace(/[^a-zA-Z0-9]/g, '');
    return alphaNum.length / trimmed.length > 0.5;
  });
  text = filtered.join('\n');

  // Enrich metadata
  const metadata = { ...artifact.metadata };
  metadata.wordCount = text.split(/\s+/).length;
  metadata.charCount = text.length;

  return {
    path: artifact.path,
    text,
    metadata,
  };
}