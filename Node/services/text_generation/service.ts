import { ServiceController } from '../../components/service';
import type { ServiceProvider } from '../../components/service';
import { NetworkScene } from 'ubiq-server/ubiq';
import nconf from 'nconf';
import { createOpenAIProvider } from './providers/openai/provider';

class TextGenerationService extends ServiceController {
    constructor(scene: NetworkScene, provider?: ServiceProvider) {
        const resolvedProvider =
            provider ??
            createOpenAIProvider({
                preprompt: nconf.get('preprompt') || '',
                promptSuffix: nconf.get('prompt_suffix') || '',
            });
        super(scene, 'TextGenerationService', resolvedProvider);
    }
}

export { TextGenerationService };
