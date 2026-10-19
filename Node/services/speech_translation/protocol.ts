import { StringDecoder } from 'node:string_decoder';

/** Streaming NDJSON decoder with bounded lines and UTF-8 chunk boundary handling. */
export class JsonLineDecoder {
    private decoder = new StringDecoder('utf8');
    private pending = '';
    private pendingBytes = 0;
    private discard = false;
    private ended = false;

    constructor(
        private readonly onMessage: (message: unknown) => void,
        private readonly onError: (error: Error) => void = () => {},
        private readonly maxLineBytes = 1024 * 1024,
    ) {
        if (!Number.isInteger(maxLineBytes) || maxLineBytes < 1) throw new Error('maxLineBytes must be positive');
    }

    push(chunk: Buffer | string): void {
        if (this.ended) throw new Error('Cannot push into an ended JSON line decoder');
        const input = typeof chunk === 'string' ? Buffer.from(chunk) : chunk;
        let start = 0;
        while (start < input.length) {
            const newline = input.indexOf(10, start);
            const end = newline === -1 ? input.length : newline;
            const fragment = input.subarray(start, end);
            if (!this.discard) {
                this.pendingBytes += fragment.length;
                if (this.pendingBytes > this.maxLineBytes) {
                    this.discard = true;
                    this.pending = '';
                    this.decoder = new StringDecoder('utf8');
                    this.onError(new Error(`JSON line exceeds ${this.maxLineBytes} bytes`));
                } else this.pending += this.decoder.write(fragment);
            }
            if (newline === -1) break;
            if (!this.discard) this.deliver();
            this.pending = '';
            this.pendingBytes = 0;
            this.discard = false;
            this.decoder = new StringDecoder('utf8');
            start = newline + 1;
        }
    }

    end(): void {
        if (this.ended) return;
        this.ended = true;
        if (!this.discard && this.pendingBytes) this.deliver();
        this.pending = '';
        this.pendingBytes = 0;
    }

    private deliver(): void {
        const line = (this.pending + this.decoder.end()).trim();
        if (!line) return;
        let message: unknown;
        try { message = JSON.parse(line); }
        catch { this.onError(new Error('Worker emitted an invalid JSON line')); return; }
        this.onMessage(message);
    }
}
