import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { Provider } from '@/components/provider';
import { baseOptions } from '@/lib/layout.shared';
import { source } from '@/lib/source';
import { appName } from '@/lib/shared';
import type { Metadata } from 'next';
import './global.css';

export const metadata: Metadata = {
  title: {
    default: appName,
    template: `%s — ${appName}`,
  },
  description: 'A sage that only speaks from your corpus.',
};

/**
 * One layout for the whole site.
 *
 * The site is unversioned and has no landing page separate from its
 * documentation — the front page *is* a document — so there is no second route
 * group and no second shell to keep in step with this one.
 */
export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="flex min-h-screen flex-col">
        <Provider>
          <DocsLayout tree={source.getPageTree()} {...baseOptions()}>
            {children}
          </DocsLayout>
        </Provider>
      </body>
    </html>
  );
}
