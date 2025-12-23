import { writeFileSync } from 'fs';
import { NormalizedArtifact } from '../normalize/index.js';
import { TokenizedChunk } from '../tokenize/index.js';

export interface DatasetManifest {
  version: string;
  created: string;
  artifacts: {
    path: string;
    checksum: string;
    chunks: string[];
  }[];
  chunks: {
    id: string;
    text: string;
    tokens: number;
    artifact: string;
  }[];
  splits: {
    train: string[];
    val: string[];
    test: string[];
  };
}

export function assembleDataset(artifacts: NormalizedArtifact[], chunks: TokenizedChunk[], outputPath: string) {
  const manifest: DatasetManifest = {
    version: '1.0',
    created: new Date().toISOString(),
    artifacts: artifacts.map(a => ({
      path: a.path,
      checksum: '', // TODO: add checksum
      chunks: chunks.filter(c => c.text.includes(a.text.slice(0, 100))).map(c => c.id), // rough match
    })),
    chunks: chunks.map(c => ({
      id: c.id,
      text: c.text,
      tokens: c.tokens.length,
      artifact: '', // TODO: map back
    })),
    splits: {
      train: chunks.slice(0, Math.floor(chunks.length * 0.7)).map(c => c.id),
      val: chunks.slice(Math.floor(chunks.length * 0.7), Math.floor(chunks.length * 0.85)).map(c => c.id),
      test: chunks.slice(Math.floor(chunks.length * 0.85)).map(c => c.id),
    },
  };

  writeFileSync(`${outputPath}/DATASET_MANIFEST.json`, JSON.stringify(manifest, null, 2));

  // Write chunks as JSONL
  const jsonl = chunks.map(c => JSON.stringify({ id: c.id, text: c.text })).join('\n');
  writeFileSync(`${outputPath}/chunks.jsonl`, jsonl);
}