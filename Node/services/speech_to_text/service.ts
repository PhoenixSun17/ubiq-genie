import { ServiceController } from '../../components/service';
import type { ServiceProvider } from '../../components/service';
import { NetworkScene } from 'ubiq-server/ubiq';
import { AzureSTTProvider } from './providers/azure/provider';

class SpeechToTextService extends ServiceController {
    constructor(scene: NetworkScene, provider: ServiceProvider = AzureSTTProvider) {
        super(scene, 'SpeechToTextService', provider);
    }
}

export { SpeechToTextService };
