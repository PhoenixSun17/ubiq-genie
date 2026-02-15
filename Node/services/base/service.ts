import { ServiceController } from '../../components/service';
import type { ServiceProvider } from '../../components/service';
import { NetworkScene } from 'ubiq-server/ubiq';
import { IntervalPrinterProvider } from './providers/interval_printer/provider';

class BaseService extends ServiceController {
    constructor(scene: NetworkScene, provider: ServiceProvider = IntervalPrinterProvider) {
        super(scene, 'BaseService', provider);
    }
}

export { BaseService };
