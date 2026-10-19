import { EventEmitter } from 'node:events';
import { watch, type FSWatcher } from 'node:fs';
import { open } from 'node:fs/promises';
import { basename, dirname, extname, resolve } from 'node:path';
import type { LogFunction, VoiceProfile } from './types';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

async function boundedRead(path: string, maxBytes: number): Promise<Buffer> {
    const file = await open(path, 'r');
    try {
        const stat = await file.stat();
        if (!stat.isFile() || stat.size > maxBytes) throw new Error(`Expected a file of at most ${maxBytes} bytes: ${path}`);
        // Read at most maxBytes+1 even if a concurrent writer grows the file.
        const chunks: Buffer[] = [];
        let size = 0;
        while (size <= maxBytes) {
            const buffer = Buffer.alloc(Math.min(65536, maxBytes + 1 - size));
            const { bytesRead } = await file.read(buffer, 0, buffer.length, null);
            if (!bytesRead) break;
            chunks.push(buffer.subarray(0, bytesRead));
            size += bytesRead;
        }
        if (size > maxBytes) throw new Error(`File exceeds ${maxBytes} bytes: ${path}`);
        return Buffer.concat(chunks, size);
    } finally { await file.close(); }
}

/** Inspect RIFF chunks without launching FFmpeg or importing an audio model. */
export function wavDuration(wav: Buffer): number {
    if (wav.length < 44 || wav.toString('ascii', 0, 4) !== 'RIFF' || wav.toString('ascii', 8, 12) !== 'WAVE') {
        throw new Error('Reference must be an uncompressed PCM or IEEE-float WAV');
    }
    const end = wav.readUInt32LE(4) + 8;
    if (end > wav.length || end < 44) throw new Error('Reference WAV is truncated');
    let byteRate = 0;
    let blockAlign = 0;
    let dataBytes = 0;
    let offset = 12;
    while (offset < end) {
        if (offset + 8 > end) throw new Error('Reference WAV has an incomplete chunk header');
        const kind = wav.toString('ascii', offset, offset + 4);
        const size = wav.readUInt32LE(offset + 4);
        const start = offset + 8;
        if (start + size > end) throw new Error('Reference WAV has a truncated chunk');
        if (kind === 'fmt ') {
            if (byteRate || size < 16) throw new Error('Reference WAV has an invalid format chunk');
            let format = wav.readUInt16LE(start);
            const channels = wav.readUInt16LE(start + 2);
            const sampleRate = wav.readUInt32LE(start + 4);
            byteRate = wav.readUInt32LE(start + 8);
            blockAlign = wav.readUInt16LE(start + 12);
            const bits = wav.readUInt16LE(start + 14);
            if (format === 0xfffe) {
                if (size < 40 || wav.readUInt16LE(start + 16) < 22 ||
                    wav.subarray(start + 26, start + 40).toString('hex') !== '000000001000800000aa00389b71') {
                    throw new Error('Reference WAV has an unsupported extensible format');
                }
                format = wav.readUInt16LE(start + 24);
            }
            if (![1, 3].includes(format) || !channels || channels > 8 || sampleRate < 8000 || sampleRate > 192000 ||
                ![8, 16, 24, 32, 64].includes(bits) || (format === 3 && ![32, 64].includes(bits)) ||
                blockAlign !== channels * bits / 8 || byteRate !== sampleRate * blockAlign) {
                throw new Error('Reference WAV has invalid or unsupported PCM format fields');
            }
        } else if (kind === 'data') dataBytes += size;
        offset = start + size + (size % 2);
    }
    if (!byteRate || !dataBytes || dataBytes % blockAlign) throw new Error('Reference WAV has no valid audio samples');
    return dataBytes / byteRate;
}

function sameProfile(left: VoiceProfile | undefined, right: VoiceProfile | undefined): boolean {
    return left?.refAudioPath === right?.refAudioPath && left?.promptText === right?.promptText &&
        left?.promptLanguage === right?.promptLanguage;
}

export class VoiceProfileStore extends EventEmitter {
    private readonly path: string;
    private profiles = new Map<string, VoiceProfile>();
    private watcher?: FSWatcher;
    private timer?: ReturnType<typeof setTimeout>;
    private closed = false;
    private pending: Promise<boolean> = Promise.resolve(true);

    constructor(path: string, private readonly log: LogFunction = console.error) {
        super();
        this.path = resolve(path);
    }

    snapshot(peerUuid: string): VoiceProfile | undefined {
        const profile = this.profiles.get(peerUuid.toLowerCase());
        return profile ? { ...profile } : undefined;
    }

    async start(): Promise<void> {
        if (this.closed) throw new Error('VoiceProfileStore is closed');
        if (this.watcher) return;
        // Watching the directory also handles an editor's atomic file replacement.
        this.watcher = watch(dirname(this.path), (_event, name) => {
            if (name !== null && name.toString().toLowerCase() !== basename(this.path).toLowerCase()) return;
            if (this.timer) clearTimeout(this.timer);
            this.timer = setTimeout(() => { this.timer = undefined; void this.reload(); }, 150);
        });
        this.watcher.on('error', (error) => this.log(`Voice profile watcher error: ${error.message}`));
        if (!await this.reload()) {
            this.close();
            throw new Error(`Initial voice profile validation failed: ${this.path}`);
        }
    }

    /** Whole-file validation is atomic; failed reloads retain the last valid mapping. */
    reload(): Promise<boolean> {
        this.pending = this.pending.then(() => this.load(), () => this.load());
        return this.pending;
    }

    close(): void {
        this.closed = true;
        if (this.timer) clearTimeout(this.timer);
        this.watcher?.close();
        this.watcher = undefined;
    }

    private async load(): Promise<boolean> {
        if (this.closed) return false;
        try {
            const raw: unknown = JSON.parse((await boundedRead(this.path, 1024 * 1024)).toString('utf8').replace(/^\uFEFF/, ''));
            if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Voice profiles must be a JSON object keyed by peer UUID');
            const entries = Object.entries(raw);
            if (entries.length > 256) throw new Error('At most 256 voice profiles may be configured');
            const next = new Map<string, VoiceProfile>();
            for (const [uuid, value] of entries) {
                if (!UUID.test(uuid) || next.has(uuid.toLowerCase())) throw new Error(`Invalid or duplicate peer UUID: ${uuid}`);
                if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`Invalid voice profile for ${uuid}`);
                const profile = value as Record<string, unknown>;
                if (typeof profile.refAudioPath !== 'string' || !profile.refAudioPath.trim() ||
                    typeof profile.promptText !== 'string' || !profile.promptText.trim() || profile.promptText.length > 16000 ||
                    (profile.promptLanguage !== 'zh' && profile.promptLanguage !== 'en')) {
                    throw new Error(`Profile ${uuid} requires refAudioPath, nonempty promptText, and promptLanguage zh/en`);
                }
                const refAudioPath = resolve(dirname(this.path), profile.refAudioPath);
                if (extname(refAudioPath).toLowerCase() !== '.wav') throw new Error(`Reference audio must use a .wav extension: ${uuid}`);
                const duration = wavDuration(await boundedRead(refAudioPath, 24 * 1024 * 1024));
                if (duration < 3 || duration > 10) throw new Error(`Reference audio for ${uuid} must last 3..10 seconds (found ${duration.toFixed(3)})`);
                next.set(uuid.toLowerCase(), {
                    refAudioPath,
                    promptText: profile.promptText.trim(),
                    promptLanguage: profile.promptLanguage,
                });
            }
            if (this.closed) return false;
            const changed = [...new Set([...this.profiles.keys(), ...next.keys()])]
                .filter((uuid) => !sameProfile(this.profiles.get(uuid), next.get(uuid)));
            this.profiles = next;
            this.log(`Loaded ${next.size} validated voice profile(s)`);
            this.emit('changed', changed);
            return true;
        } catch (error) {
            this.log(`Voice profile reload rejected; keeping last valid configuration: ${error instanceof Error ? error.message : String(error)}`);
            return false;
        }
    }
}
