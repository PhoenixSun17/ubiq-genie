import type { ServiceProvider } from '../../../../../components/service';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const generateScript = path.join(__dirname, '..', 'generate.mjs');

/**
 * Qwen3-4B via ollama (GGUF Q8_0 from HuggingFace).
 * Requires `ollama serve` to be running.
 * Set `thinking: true` to enable Qwen3's internal reasoning mode.
 * Default is off (appends /no_think) for low-latency voice use.
 */
export function createOllamaQwen3_4BProvider(options?: {
    preprompt?: string;
    thinking?: boolean;
    host?: string;
}): ServiceProvider {
    const args = [
        generateScript,
        '--model', 'hf.co/Qwen/Qwen3-4B-GGUF:Q8_0',
    ];
    if (options?.preprompt) {
        args.push('--preprompt', options.preprompt);
    }
    if (options?.thinking) {
        args.push('--thinking');
    }
    if (options?.host) {
        args.push('--host', options.host);
    }
    return {
        name: 'ollama-qwen3-4b',
        command: 'node',
        args,
        processMode: 'singleton',
    };
}
