export class Logger {
  static info(msg: string) {
    console.log(`[INFO] ${msg}`);
  }

  static error(msg: string) {
    console.error(`[ERROR] ${msg}`);
  }
}