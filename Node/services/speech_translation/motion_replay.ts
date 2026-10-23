import { randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { performance } from 'node:perf_hooks';
import { NetworkId, type NetworkScene } from '@ucl-vr/ubiq';

export const MOTION_REPLAY_NETWORK_ID = 101;
export const MOTION_REPLAY_PROPERTY = 'ubiq.genie.sovits.motion';
export type MotionReplayTerminalStatus = 'sent' | 'cancelled' | 'skipped' | 'failed';

interface UtteranceIdentity {
    requestId: string;
    speakerUuid: string;
}

export type MotionReplayLifecycle =
    | (UtteranceIdentity & { type: 'accepted'; sourceStartMs: number; sourceEndMs: number })
    | (UtteranceIdentity & { type: 'play'; audioLength: number; sampleRate: number; sampleCount: number })
    | (UtteranceIdentity & { type: 'terminal'; status: MotionReplayTerminalStatus });

interface ReplayRoom extends EventEmitter {
    peer: { uuid: string; setProperty(key: string, value: string): void };
    peers: Map<string, { uuid: string }>;
    room: { uuid: string };
}

const MAX_MESSAGE_BYTES = 2048;
const MAX_PENDING_TERMINALS = 32;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Optional scene capability: old scenes ignore ID 101 and retain ID 95 playback.
 * All timing is in Node performance.now() milliseconds, scoped to sessionId.
 */
export class MotionReplayController {
    readonly networkId = new NetworkId(MOTION_REPLAY_NETWORK_ID);
    private session = randomUUID();
    get sessionId(): string { return this.session; }
    private closed = false;
    private joined = false;
    private readonly lastPing = new Map<string, number>();
    private readonly pendingTerminals = new Map<string, MotionReplayLifecycle>();
    private readonly retryTimer: NodeJS.Timeout;

    constructor(
        private readonly scene: NetworkScene,
        private readonly room: ReplayRoom,
        private readonly service: EventEmitter,
        private readonly outputSampleRate: number,
        private readonly log: (message: string) => void,
    ) {
        scene.register(this);
        service.on('motionReplay', this.onLifecycle);
        service.on('translationClosed', this.onServiceClosed);
        room.on('OnJoinedRoom', this.onJoined);
        room.on('OnPeerRemoved', this.onPeerRemoved);
        this.retryTimer = setInterval(() => this.flushTerminals(), 250);
        this.retryTimer.unref();
        if (UUID.test(room.room.uuid)) this.onJoined();
    }

    private readonly onJoined = (): void => {
        if (this.closed) return;
        this.joined = UUID.test(this.room.room.uuid);
        this.lastPing.clear();
        if (this.joined) {
            this.room.peer.setProperty(MOTION_REPLAY_PROPERTY, JSON.stringify({ version: 1, sessionId: this.sessionId }));
        }
    };

    private readonly onPeerRemoved = (peer: { uuid: string }): void => {
        this.lastPing.delete(peer.uuid);
    };

    private readonly onServiceClosed = (): void => this.close();

    private readonly onLifecycle = (event: MotionReplayLifecycle): void => {
        this.flushTerminals();
        if (event.type === 'play') {
            // Reserve the following AudioInfo framing too. Both writes happen
            // synchronously on the same ordered connection, without awaiting.
            const header = { type: 'AudioInfo', targetPeer: '', audioLength: String(event.audioLength) };
            this.send(event, Buffer.byteLength(JSON.stringify(header)) + 12);
            return;
        }
        try {
            this.send(event);
        } catch (error) {
            if (event.type === 'terminal') {
                if (!this.pendingTerminals.has(event.requestId) && this.pendingTerminals.size >= MAX_PENDING_TERMINALS) {
                    // A changed advertised session invalidates all client pins.
                    // This is an exceptional recovery for prolonged congestion.
                    this.session = randomUUID();
                    this.pendingTerminals.clear();
                    this.onJoined();
                    this.log('Motion terminal retry capacity reached; replay session reset.');
                } else {
                    this.pendingTerminals.set(event.requestId, event);
                }
            }
            this.log(`Motion ${event.type} metadata unavailable: ${error instanceof Error ? error.message : String(error)}`);
        }
    };

    private flushTerminals(): void {
        if (this.closed || !this.joined) return;
        for (const [requestId, event] of this.pendingTerminals) {
            try { this.send(event); } catch { break; }
            this.pendingTerminals.delete(requestId);
        }
    }

    processMessage(message: { message: Buffer }): void {
        const s1 = performance.now();
        if (this.closed || !this.joined || !Buffer.isBuffer(message.message) ||
            message.message.length > MAX_MESSAGE_BYTES) return;
        let parsed: unknown;
        try { parsed = JSON.parse(message.message.toString('utf8')); } catch { return; }
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return;
        const ping = parsed as Record<string, unknown>;
        if (ping.version !== 1 || ping.type !== 'ping' || ping.backendUuid !== this.room.peer.uuid ||
            ping.sessionId !== this.sessionId || typeof ping.peerUuid !== 'string' ||
            !this.room.peers.has(ping.peerUuid) || typeof ping.nonce !== 'string' ||
            ping.nonce.length < 1 || ping.nonce.length > 128 ||
            typeof ping.t0 !== 'number' || !Number.isFinite(ping.t0) || ping.t0 < 0) return;
        const previous = this.lastPing.get(ping.peerUuid);
        if (previous !== undefined && s1 - previous < 50) return;
        this.lastPing.set(ping.peerUuid, s1);
        try {
            this.send({
                type: 'pong', peerUuid: ping.peerUuid, nonce: ping.nonce, t0: ping.t0,
                s1, s2: performance.now(),
            });
        } catch {
            // Clock estimation retries; congestion must not grow a response queue.
        }
    }

    private send(event: object, reservedBytes = 0): void {
        if (this.closed || !this.joined) throw new Error('translation backend is not in a room');
        const payload = {
            version: 1, backendUuid: this.room.peer.uuid, sessionId: this.sessionId, ...event,
        };
        const bytes = Buffer.byteLength(JSON.stringify(payload));
        if (bytes > MAX_MESSAGE_BYTES) throw new Error('metadata exceeds its message limit');
        if (!this.scene.connections.length) throw new Error('no Ubiq connection');
        for (const connection of this.scene.connections) {
            const socket = (connection as unknown as { socket?: {
                writableLength: number; destroyed: boolean; writable: boolean;
            } }).socket;
            if (!socket || socket.destroyed || !socket.writable || !Number.isFinite(socket.writableLength)) {
                throw new Error('metadata requires a writable Ubiq TCP connection');
            }
            if (socket.writableLength + bytes + 12 + reservedBytes > this.outputSampleRate * 2) {
                throw new Error('Ubiq TCP output is congested');
            }
        }
        this.scene.send(this.networkId, payload);
    }

    close(): void {
        if (this.closed) return;
        this.closed = true;
        this.service.off('motionReplay', this.onLifecycle);
        this.service.off('translationClosed', this.onServiceClosed);
        this.room.off('OnJoinedRoom', this.onJoined);
        this.room.off('OnPeerRemoved', this.onPeerRemoved);
        this.scene.unregister(this);
        clearInterval(this.retryTimer);
        this.pendingTerminals.clear();
        this.lastPing.clear();
        if (this.joined) {
            try { this.room.peer.setProperty(MOTION_REPLAY_PROPERTY, ''); } catch { /* Connection may already be closed. */ }
        }
        this.joined = false;
    }
}
