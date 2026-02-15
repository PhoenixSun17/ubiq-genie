import { ServiceController } from '../../components/service';
import type { ServiceProvider } from '../../components/service';
import { NetworkScene } from 'ubiq-server/ubiq';
import { createStableDiffusionProvider } from './providers/stable_diffusion/provider';

class ImageGenerationService extends ServiceController {
    constructor(
        scene: NetworkScene,
        provider?: ServiceProvider,
        options?: { outputFolder?: string; promptPostfix?: string }
    ) {
        const resolvedProvider =
            provider ??
            createStableDiffusionProvider({
                outputFolder: options?.outputFolder ?? '../../apps/texture_generation/data',
                promptPostfix: options?.promptPostfix ?? ', 4k',
            });
        super(scene, 'ImageGenerationService', resolvedProvider);
    }
}

export { ImageGenerationService };
