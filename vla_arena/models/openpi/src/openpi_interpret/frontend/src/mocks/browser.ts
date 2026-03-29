import { setupWorker } from 'msw/browser';
import { handlers } from './handlers';

/** MSW service worker instance for browser-based API mocking. */
export const worker = setupWorker(...handlers);
