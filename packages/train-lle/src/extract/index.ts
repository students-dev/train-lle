import { readFileSync } from 'fs';
import * as pdfParse from 'pdf-parse';
import * as mammoth from 'mammoth';
import * as cheerio from 'cheerio';
import { createWorker } from 'tesseract.js';
import { simpleParser } from 'mailparser';
import sharp from 'sharp';

export interface ExtractedArtifact {
  path: string;
  text: string;
  metadata: Record<string, any>;
}

export async function extractText(artifact: { path: string; mimeType: string }): Promise<ExtractedArtifact> {
  const buffer = readFileSync(artifact.path);
  let text = '';
  const metadata: Record<string, any> = {};

  if (artifact.mimeType === 'application/pdf') {
    const data = await pdfParse(buffer);
    text = data.text;
    metadata.pages = data.numpages;
  } else if (artifact.mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
    const result = await mammoth.extractRawText({ buffer });
    text = result.value;
  } else if (artifact.mimeType.startsWith('text/html')) {
    const $ = cheerio.load(buffer.toString());
    text = $('body').text();
  } else if (artifact.mimeType.startsWith('image/')) {
    const worker = await createWorker();
    const { data: { text: ocrText } } = await worker.recognize(buffer);
    await worker.terminate();
    text = ocrText;
  } else if (artifact.mimeType === 'message/rfc822') {
    const parsed = await simpleParser(buffer);
    text = `${parsed.subject}\n${parsed.text}`;
    metadata.from = parsed.from?.text;
    metadata.to = parsed.to?.text;
  } else if (artifact.mimeType.startsWith('text/') || artifact.mimeType === 'application/json') {
    text = buffer.toString();
  } else {
    // For other types, try as text
    text = buffer.toString();
  }

  return {
    path: artifact.path,
    text,
    metadata,
  };
}