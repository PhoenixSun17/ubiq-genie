import { ServiceController } from '../../components/service';
import type { ServiceProvider } from '../../components/service';
import { NetworkScene } from 'ubiq-server/ubiq';
import { createPersonaPlexProvider } from './providers/personaplex/provider';

/**
 * Audio-to-audio service that manages a child process capable of ingesting
 * raw audio and producing raw audio (and optionally text) output.
 *
 * The default provider is PersonaPlex, a full-duplex speech-to-speech model
 * that replaces the traditional STT → text generation → TTS pipeline.
 *
 * The child process communicates via a binary length-prefixed framing protocol
 * on stdin/stdout. Consumers (e.g. the conversational agent app) are responsible
 * for encoding/decoding the framed packets using the protocol utilities from
 * `./providers/personaplex/protocol.ts`.
 */
class AudioToAudioService extends ServiceController {
    constructor(scene: NetworkScene, provider?: ServiceProvider) {
        const resolvedProvider = provider ?? createPersonaPlexProvider();
        super(scene, 'AudioToAudioService', resolvedProvider);
    }
}

export { AudioToAudioService };
