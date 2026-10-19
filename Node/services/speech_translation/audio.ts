import { spawn } from 'node:child_process';
import type { AudioOptions } from './types';

/** Split without padding, truncation, or a tiny final packet that Unity treats as a header. */
export function splitPcm(pcm: Buffer, maxBytes = 16000, minBytes = 200): Buffer[] {
    if (!Number.isInteger(maxBytes) || !Number.isInteger(minBytes) || maxBytes % 2 || minBytes % 2 ||
        minBytes < 2 || maxBytes < minBytes || pcm.length % 2 || pcm.length < minBytes) {
        throw new Error('PCM and packet sizes must be even, with enough audio for the minimum packet size');
    }
    const count = Math.ceil(pcm.length / maxBytes);
    if (pcm.length < count * minBytes) throw new Error('PCM cannot fit the requested packet size bounds');
    const chunks: Buffer[] = [];
    let offset = 0;
    for (let index = 0; index < count; index++) {
        const remainingPackets = count - index - 1;
        const size = Math.min(maxBytes, pcm.length - offset - remainingPackets * minBytes);
        chunks.push(pcm.subarray(offset, offset + size));
        offset += size;
    }
    return chunks;
}

export function wavToPcm(wav: Buffer, options: AudioOptions, signal: AbortSignal): Promise<Buffer> {
    if (!Number.isInteger(options.sampleRate) || options.sampleRate < 8000 || options.sampleRate > 96000) {
        throw new Error('Output sampleRate must be an integer from 8000 to 96000');
    }
    if (!options.ffmpegPath || !Number.isInteger(options.timeoutMs) || options.timeoutMs <= 0) {
        throw new Error('FFmpeg path and a positive integer timeoutMs are required');
    }
    if (wav.length < 44 || wav.length > 32 * 1024 * 1024 ||
        wav.toString('ascii', 0, 4) !== 'RIFF' || wav.toString('ascii', 8, 12) !== 'WAVE' ||
        wav.readUInt32LE(4) + 8 > wav.length) {
        throw new Error('Expected a WAV of at most 32 MiB');
    }
    return new Promise((resolve, reject) => {
        if (signal.aborted) { reject(new Error('Audio conversion cancelled')); return; }
        const child = spawn(options.ffmpegPath, [
            '-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'wav', '-i', 'pipe:0',
            '-vn', '-ac', '1', '-ar', String(options.sampleRate), '-acodec', 'pcm_s16le', '-f', 's16le', 'pipe:1',
        ], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
        const chunks: Buffer[] = [];
        const maxBytes = options.sampleRate * 2 * 120;
        let bytes = 0;
        let stderr = '';
        let finished = false;
        let timeout: ReturnType<typeof setTimeout> | undefined;
        const finish = (error?: Error) => {
            if (finished) return;
            finished = true;
            if (timeout) clearTimeout(timeout);
            signal.removeEventListener('abort', abort);
            if (error) { child.kill(); reject(error); }
            else resolve(Buffer.concat(chunks, bytes));
        };
        const abort = () => finish(new Error('Audio conversion cancelled'));
        signal.addEventListener('abort', abort, { once: true });
        timeout = setTimeout(() => finish(new Error('FFmpeg conversion timed out')), options.timeoutMs);
        child.on('error', (error) => finish(error));
        child.stdin.on('error', (error) => finish(error));
        child.stdout.on('data', (chunk: Buffer) => {
            if (finished) return;
            bytes += chunk.length;
            if (bytes > maxBytes) finish(new Error('Synthesized audio exceeds 120 seconds'));
            else chunks.push(chunk);
        });
        child.stderr.on('data', (chunk: Buffer) => { stderr = (stderr + chunk.toString('utf8')).slice(-8192); });
        child.on('close', (code) => {
            if (code !== 0) finish(new Error(`FFmpeg exited with ${code}: ${stderr.trim()}`));
            else if (bytes < 200 || bytes % 2) finish(new Error('FFmpeg returned empty, too short, or unaligned PCM16 audio'));
            else finish();
        });
        child.stdin.end(wav);
    });
}
