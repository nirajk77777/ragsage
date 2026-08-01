import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';

export const revalidate = false;

/**
 * A prebuilt Orama index, exported as a static file and searched in the browser.
 *
 * `staticGET` rather than `GET`: the production image has no runtime, so search
 * cannot be a server that answers queries. It indexes prose and the generated API
 * pages alike, because they are one content source — a reader looking for a
 * symbol should not have to know which page holds it.
 */
export const { staticGET: GET } = createFromSource(source, {
  language: 'english',
});
