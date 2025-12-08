import { Tensor } from "../core/tensor";

export class ImageLoader {
  static load(path: string): Tensor {
    // Placeholder: assume raw bytes, reshape to [1, H, W]
    // In real, use sharp or canvas to load image.
    // For MVP, throw error.
    throw new Error("Image loading not implemented without sharp");
  }
}