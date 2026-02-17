import type { ServiceProvider } from '../../../../../components/service';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const generateScript = path.join(__dirname, '..', 'generate.mjs');

/**
 * Qwen3-4B via llama.cpp (node-llama-cpp, GGUF Q8_0).
 * Set `thinking: true` to enable Qwen3's internal reasoning mode.
 * Default is off (appends /no_think) for low-latency voice use.
 */
export function createLlamaCppQwen3_4BProvider(options?: {
    preprompt?: string;
    thinking?: boolean;
}): ServiceProvider {
    const args = [
        generateScript,
        '--model', 'hf:Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf',
    ];
    if (options?.preprompt) {
        args.push('--preprompt', options.preprompt);
    }
    if (options?.thinking) {
        args.push('--thinking');
    }
    return {
        name: 'llama-cpp-qwen3-4b',
        command: 'node',
        args,
        processMode: 'singleton',
    };
}
