import { EventEmitter } from 'node:events';
import type { VoiceCatalogEntry } from './types';

export const VOICE_CATALOG_PROPERTY = 'ubiq.genie.sovits.catalog';
export const VOICE_SELECTION_PROPERTY = 'ubiq.genie.sovits.selection';
export const VOICE_BINDINGS_PROPERTY = 'ubiq.genie.sovits.bindings';
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const VOICE_ID = /^[0-9a-f]{64}$/;
const MAX_SELECTION_BYTES = 1024;
const MAX_METADATA_BYTES = 128 * 1024;
const MAX_BINDINGS = 256;

interface Peer {
    uuid: string;
    properties: Map<string, string>;
}

interface BindingRoom extends EventEmitter {
    peer: Peer & { setProperty(key: string, value: string): void };
    peers: Map<string, Peer>;
    room: { uuid: string };
}

interface BindingService extends EventEmitter {
    listVoices(): VoiceCatalogEntry[];
    bindVoice(peerUuid: string, voiceId: string): void;
    getBoundVoiceId(peerUuid: string): string | undefined;
}

interface Selection {
    version: 1;
    requestId: string;
    voiceId: string;
    backendUuid: string;
}

interface Binding {
    peerUuid: string;
    requestId: string;
    voiceId: string;
    status: 'bound' | 'error';
    message: string;
}

/** Publishes metadata only. Reference material never enters room properties. */
export class VoiceBindingController {
    private joined = false;
    private closed = false;
    private readonly seen = new Map<string, string>();
    private readonly bindings = new Map<string, Binding>();
    private readonly accepted = new Set<string>();
    private readonly published = new Map<string, string>();

    constructor(
        private readonly room: BindingRoom,
        private readonly service: BindingService,
        private readonly log: (message: string) => void,
    ) {
        room.on('OnJoinedRoom', this.onJoined);
        room.on('OnPeerAdded', this.onPeer);
        room.on('OnPeerUpdated', this.onPeer);
        room.on('OnPeerRemoved', this.onRemoved);
        service.on('voiceCatalogChanged', this.onCatalogChanged);
        if (UUID.test(room.room.uuid)) this.onJoined();
    }

    close(): void {
        if (this.closed) return;
        this.closed = true;
        this.room.off('OnJoinedRoom', this.onJoined);
        this.room.off('OnPeerAdded', this.onPeer);
        this.room.off('OnPeerUpdated', this.onPeer);
        this.room.off('OnPeerRemoved', this.onRemoved);
        this.service.off('voiceCatalogChanged', this.onCatalogChanged);
        this.seen.clear();
        this.bindings.clear();
        this.accepted.clear();
        this.published.clear();
    }

    private readonly onJoined = (): void => {
        if (this.closed) return;
        this.joined = UUID.test(this.room.room.uuid);
        this.seen.clear();
        this.bindings.clear();
        this.accepted.clear();
        this.published.clear();
        if (!this.joined) return;
        this.publishCatalog();
        this.publishBindings();
        // Existing peer properties also carry the choice when Node joins later.
        for (const peer of this.room.peers.values()) this.onPeer(peer);
    };

    private readonly onPeer = (peer: Peer | undefined): void => {
        if (this.closed || !this.joined || !peer || peer.uuid === this.room.peer.uuid ||
            !this.room.peers.has(peer.uuid) || !UUID.test(peer.uuid)) return;
        const raw = peer.properties.get(VOICE_SELECTION_PROPERTY);
        if (!raw || this.seen.get(peer.uuid) === raw) return;
        if (Buffer.byteLength(raw, 'utf8') > MAX_SELECTION_BYTES) {
            this.log(`Ignored oversized voice selection from ${peer.uuid}.`);
            return;
        }
        let parsed: unknown;
        try { parsed = JSON.parse(raw); }
        catch { this.log(`Ignored malformed voice selection from ${peer.uuid}.`); return; }
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return;
        const value = parsed as Record<string, unknown>;
        if (value.version !== 1 || typeof value.requestId !== 'string' || !UUID.test(value.requestId) ||
            typeof value.voiceId !== 'string' || !VOICE_ID.test(value.voiceId) ||
            typeof value.backendUuid !== 'string' || value.backendUuid !== this.room.peer.uuid) return;
        if (!this.bindings.has(peer.uuid) && this.bindings.size >= MAX_BINDINGS) {
            this.log(`Ignored voice selection from ${peer.uuid}: binding metadata capacity reached.`);
            return;
        }
        const selection = value as unknown as Selection;
        this.seen.set(peer.uuid, raw);
        this.accepted.delete(peer.uuid);
        const binding: Binding = {
            peerUuid: peer.uuid,
            requestId: selection.requestId,
            voiceId: selection.voiceId,
            status: 'error',
            message: 'Voice selection could not be applied.',
        };
        if (!this.service.listVoices().some((voice) => voice.id === selection.voiceId)) {
            binding.message = 'Voice is unavailable. Refresh the list and choose another voice.';
        } else {
            try {
                // User identity comes from the peer event, never a UUID in the payload.
                this.service.bindVoice(peer.uuid, selection.voiceId);
                if (this.service.getBoundVoiceId(peer.uuid) === selection.voiceId) {
                    binding.status = 'bound';
                    binding.message = 'Voice applied. New utterances use this voice.';
                    this.accepted.add(peer.uuid);
                }
            } catch (error) {
                // Backend errors may contain local file paths: do not publish them.
                this.log(`Voice binding rejected for ${peer.uuid}: ${error instanceof Error ? error.message : String(error)}`);
                binding.message = 'Voice selection could not be applied. Check the Node log.';
            }
        }
        this.bindings.set(peer.uuid, binding);
        this.log(`Voice binding ${binding.status} for ${peer.uuid}: ${binding.message}`);
        this.publishBindings();
    };

    private readonly onRemoved = (peer: Peer): void => {
        this.seen.delete(peer.uuid);
        this.accepted.delete(peer.uuid);
        if (this.bindings.delete(peer.uuid)) this.publishBindings();
    };

    private readonly onCatalogChanged = (): void => {
        if (this.closed || !this.joined) return;
        this.publishCatalog();
        for (const binding of this.bindings.values()) {
            if (!this.accepted.has(binding.peerUuid)) continue;
            if (this.service.getBoundVoiceId(binding.peerUuid) === binding.voiceId) {
                binding.status = 'bound';
                binding.message = 'Voice applied. New utterances use this voice.';
            } else {
                binding.status = 'error';
                binding.message = 'Selected voice is unavailable. Choose a voice from the current list.';
            }
        }
        this.publishBindings();
    };

    private publishCatalog(): void {
        // Explicit projection prevents future private profile fields leaking.
        const voices = this.service.listVoices().map(({ id, name, language }) => ({ id, name, language }));
        this.publish(VOICE_CATALOG_PROPERTY, { version: 1, voices });
    }

    private publishBindings(): void {
        this.publish(VOICE_BINDINGS_PROPERTY, {
            version: 1,
            bindings: [...this.bindings.values()].sort((a, b) => a.peerUuid.localeCompare(b.peerUuid)),
        });
    }

    private publish(key: string, payload: object): void {
        if (this.closed || !this.joined) return;
        const value = JSON.stringify(payload);
        if (Buffer.byteLength(value, 'utf8') > MAX_METADATA_BYTES) {
            this.log('Voice selector metadata exceeds its size limit; update was not published.');
            return;
        }
        if (this.published.get(key) === value) return;
        this.room.peer.setProperty(key, value);
        this.published.set(key, value);
    }
}
