import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ApplicationController } from '../../components/application';
import { VoipReceiver } from '../../components/voip_receiver';
import { SpeechTranslationService } from '../../services/speech_translation/service';
import { readTranslationConfig } from '../../services/speech_translation/config';

class GptSovitsTranslation extends ApplicationController {
    private translation!: SpeechTranslationService;

    async start(): Promise<void> {
        this.registerComponents();
        this.definePipeline();
        await this.translation.start();
        await this.joinRoom();
        this.connection.onClose.push(() => {
            this.log('Ubiq connection closed; stopping translation and discarding pending speech.', 'warning');
            this.translation.close();
            process.exit(1);
        });
        this.log('Waiting for peers. Bind their current UUIDs in profiles.json to enable translation.');
    }

    registerComponents(): void {
        this.translation = new SpeechTranslationService(this.scene, readTranslationConfig(this.configPath));
        this.components.translation = this.translation;
        this.components.voipReceiver = new VoipReceiver(this.scene);
    }

    definePipeline(): void {
        this.components.voipReceiver.on('audio', (uuid: string, data: {
            samples: Int16Array; sampleRate: number; channelCount: number; bitsPerSample: number;
        }) => {
            if (this.roomClient.peers.has(uuid)) this.translation.acceptAudio(uuid, data);
        });
        const addPeer = (peer: { uuid: string; properties?: Map<string, string> }) => {
            if (peer.uuid === this.roomClient.peer.uuid) return;
            this.log(`Peer UUID: ${peer.uuid}; name: ${peer.properties?.get('ubiq.displayname') ?? '(unnamed)'}`);
            this.translation.addPeer(peer.uuid);
        };
        this.roomClient.on('OnPeerAdded', addPeer);
        this.roomClient.on('OnPeerRemoved', (peer: { uuid: string }) => {
            this.log(`Peer left: ${peer.uuid}`);
            this.translation.removePeer(peer.uuid);
        });
        this.roomClient.on('OnJoinRejected', (reason: unknown) => {
            this.log(`Ubiq room join rejected: ${String(reason)}`, 'error');
            this.translation.close();
            process.exit(1);
        });
        for (const peer of this.roomClient.peers.values()) addPeer(peer);
        this.translation.on('fatal', (error: Error) => {
            this.log(error.message, 'error');
            this.translation.close();
            process.exit(1);
        });
    }
}

export { GptSovitsTranslation };

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
    const app = new GptSovitsTranslation(path.join(path.dirname(fileURLToPath(import.meta.url)), 'config.json'));
    app.start().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : String(error));
        process.exit(1);
    });
}
