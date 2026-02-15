import { ServiceController } from '../../components/service';
import type { ServiceProvider } from '../../components/service';
import { NetworkScene } from 'ubiq-server/ubiq';
import { AzureTTSProvider } from './providers/azure/provider';

export class TextToSpeechService extends ServiceController {
    constructor(scene: NetworkScene, provider: ServiceProvider = AzureTTSProvider) {
        super(scene, 'TextToSpeechService', provider);
    }
}
