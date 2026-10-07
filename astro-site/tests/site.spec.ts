import { test, expect } from '@playwright/test';
const notebooks = ['byot','ecg-foundation-model','train-arrhythmia-model','train-ecg-denoiser','train-ecg-segmentation'];
test('landing leads to installation on mobile',async({page})=>{
 await page.setViewportSize({width:390,height:844});await page.goto('');
 await expect(page.locator('h1')).toHaveCount(1);
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
 await page.getByRole('link',{name:'Get started',exact:true}).first().click();
 await expect(page).toHaveURL(/\/heartkit\/quickstart\/$/);
 await page.getByRole('tab',{name:'Git clone',exact:true}).click();
 await expect(page.getByRole('tabpanel',{name:'Git clone'})).toContainText('uv sync');
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
});
test('all saved notebooks retain downloadable sources and figures',async({page,request})=>{
 for(const name of notebooks){await page.goto(`guides/${name}/`);await expect(page.locator('h1')).toHaveCount(1);
 const link=page.getByRole('link',{name:'Download notebook',exact:true});expect((await request.get((await link.getAttribute('href'))!)).ok()).toBeTruthy();
 for(const img of await page.locator('img[src*="/notebooks/"]').all())expect(await img.evaluate((e:HTMLImageElement)=>e.complete&&e.naturalWidth>0)).toBeTruthy();}
});
test('public API catalog filters results',async({page})=>{
 await page.goto('reference/');await page.getByRole('searchbox',{name:'Search Python API'}).fill('Icentia');
 await expect(page.getByRole('link',{name:'IcentiaDataset',exact:true}).first()).toBeVisible();
});
test('task result tables and charts render',async({page})=>{
 for(const task of ['denoise','segmentation','rhythm','beat']){await page.goto(`tasks/${task}/`);await expect(page.locator('.sl-markdown-content table').first()).toBeVisible();}
 await page.goto('tasks/denoise/');await expect(page.locator('iframe').first()).toBeVisible();
});
test('section navigation follows existing URLs',async({page})=>{
 await page.setViewportSize({width:1440,height:1000});
 for(const [route,label] of [['datasets/icentia11k/','User guide'],['tasks/beat/','Tasks'],['zoo/arr-2-eff-sm/','Reference']]){
 await page.goto(route);await expect(page.locator('[data-helia-sidebar-heading]')).toHaveText(label);}
});

test('mobile menu scopes pages to the active section', async ({page}) => {
 await page.setViewportSize({width:390,height:844});
 await page.goto('modes/');
 await page.getByRole('button',{name:'Open navigation',exact:true}).click();
 const sidebar = page.locator('#starlight__sidebar');
 await expect(sidebar.locator('[data-helia-sidebar-heading]')).toHaveText('User guide');
 await expect(sidebar.getByRole('link',{name:'Train a model',exact:true})).toBeVisible();
 await expect(sidebar.getByRole('link',{name:'Install heartKIT',exact:true})).toHaveCount(0);
 await page.getByRole('button',{name:'Open navigation',exact:true}).click();
 await page.locator('[data-helia-section-dropdown] summary').click();
 await page.getByRole('navigation',{name:'Choose section'}).getByRole('link',{name:'Getting started',exact:true}).click();
 await page.getByRole('button',{name:'Open navigation',exact:true}).click();
 await expect(sidebar.getByRole('link',{name:'Install heartKIT',exact:true})).toBeVisible();
 await expect(sidebar.getByRole('link',{name:'Train a model',exact:true})).toHaveCount(0);
 const title = page.locator('.helia-site-header__title');
 expect(await title.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
});
test('workflow configurations start collapsed and download as JSON', async ({page,request}) => {
 for (const route of ['train','evaluate','export']) {
  await page.goto(`modes/${route}/`);
  const example=page.locator('.config-example').first();
  await expect(example.locator('.config-preview')).toBeVisible();
  await expect(example.locator('details')).not.toHaveAttribute('open');
  const response=await request.get((await example.getByRole('link',{name:'Download'}).getAttribute('href'))!);
  expect(response.ok()).toBeTruthy(); expect(await response.json()).toHaveProperty('datasets');
  await example.locator('summary').click();await expect(example.locator('details')).toHaveAttribute('open','');
 }
});

test('legacy API links preserve symbol anchors and query strings', async ({page}) => {
 const symbol = 'heartkit.datasets.dataset.HKDataset';
 await page.goto(`api/heartkit/datasets/dataset/?from=legacy#${symbol}`);
 await expect(page).toHaveURL(new RegExp(`/reference/api/heartkit/datasets/dataset/\\?from=legacy#${symbol}$`));
 await expect(page.locator(`[id="${symbol}"]`)).toBeVisible();
});
