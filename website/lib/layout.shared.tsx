import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { Logo } from '@/components/logo';
import { appName, repoUrl } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      // The mark inherits the header's colour rather than carrying its own, so
      // it reads in both themes from one definition. Fumadocs renders this node
      // inside the link home, which is why the name stays part of it.
      title: (
        <span className="inline-flex items-center gap-2">
          <Logo />
          {appName}
        </span>
      ),
    },
    githubUrl: repoUrl,
  };
}
