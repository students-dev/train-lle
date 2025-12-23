import { glob } from 'glob';
import { readFileSync } from 'fs';
import { createHash } from 'crypto';
import { lookup } from 'mime';
import { franc } from 'franc';

export interface FileArtifact {
  path: string;
  mimeType: string;
  checksum: string;
  language: string;
  size: number;
}

export async function discoverFiles(path: string, extensions?: string[]): Promise<FileArtifact[]> {
  const patterns = extensions ? extensions.map(ext => `**/*.${ext}`) : ['**/*'];
  const files = await glob(patterns, { cwd: path, absolute: true, nodir: true });
  const artifacts: FileArtifact[] = [];

  for (const file of files) {
    const content = readFileSync(file);
    const checksum = createHash('sha256').update(content).digest('hex');
    const mimeType = lookup(file) || 'application/octet-stream';
    const language = franc(content.toString()) || 'und';
    const size = content.length;

    artifacts.push({
      path: file,
      mimeType,
      checksum,
      language,
      size,
    });
  }

  // Deduplicate by checksum
  const unique = new Map<string, FileArtifact>();
  for (const artifact of artifacts) {
    if (!unique.has(artifact.checksum)) {
      unique.set(artifact.checksum, artifact);
    }
  }

  return Array.from(unique.values());
}