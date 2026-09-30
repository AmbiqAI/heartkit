import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { heliaFrontmatterSchema } from '@ambiqai/helia-ui/starlight';

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema({ extend: heliaFrontmatterSchema }) }),
};
