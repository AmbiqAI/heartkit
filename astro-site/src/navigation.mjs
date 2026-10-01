import apiSidebar from "./data/api-sidebar.json" with { type: "json" };
const page = (label, slug) => ({ label, slug });
const group = (label, items) => ({ label, items, collapsed: true });
const apiGroups = Object.entries(Object.groupBy(apiSidebar, (item) => item.label.split(".")[1] || "Package")).map(([label, items]) => ({label, items, collapsed:true}));
export const sections = [
 {label:"Home",href:"/heartkit/",sidebar:false},
 {label:"Getting started",href:"/heartkit/quickstart/",sidebar:[page("Install heartKIT","quickstart"),page("Use the command line","usage/cli"),page("Use Python","usage/python")]},
 {label:"User guide",href:"/heartkit/guides/",sidebar:[page("Guides and examples","guides"),group("Model workflow",[page("Workflow overview","modes"),page("Configure a task","modes/configuration"),page("Download datasets","modes/download"),page("Train a model","modes/train"),page("Evaluate a model","modes/evaluate"),page("Export a model","modes/export"),page("Run a demo","modes/demo")]),group("Datasets",[
  {
    "label": "Datasets",
    "slug": "datasets"
  },
  {
    "label": "Icentia11k",
    "slug": "datasets/icentia11k"
  },
  {
    "label": "Synthetic",
    "slug": "datasets/synthetic"
  },
  {
    "label": "QTDB",
    "slug": "datasets/qtdb"
  },
  {
    "label": "LUDB",
    "slug": "datasets/ludb"
  },
  {
    "label": "LSAD",
    "slug": "datasets/lsad"
  },
  {
    "label": "PTB-XL",
    "slug": "datasets/ptbxl"
  },
  {
    "label": "MIT-BIH",
    "slug": "datasets/mitbih"
  },
  {
    "label": "BYOD",
    "slug": "datasets/byod"
  }
]),group("Models",[
  {
    "label": "Models",
    "slug": "models"
  },
  {
    "label": "BYOM",
    "slug": "models/byom"
  }
]),group("Examples",[
 page("Custom task","guides/byot"),page("Arrhythmia training","guides/train-arrhythmia-model"),page("ECG denoising","guides/train-ecg-denoiser"),page("ECG segmentation","guides/train-ecg-segmentation"),page("ECG foundation model","guides/ecg-foundation-model")])]},
 {label:"Tasks",href:"/heartkit/tasks/",sidebar:[
  {
    "label": "Tasks",
    "slug": "tasks"
  },
  {
    "label": "Denoise",
    "slug": "tasks/denoise"
  },
  {
    "label": "Segmentation",
    "slug": "tasks/segmentation"
  },
  {
    "label": "Rhythm",
    "slug": "tasks/rhythm"
  },
  {
    "label": "Beat",
    "slug": "tasks/beat"
  },
  {
    "label": "BYOT",
    "slug": "tasks/byot"
  }
]},
 {label:"Reference",href:"/heartkit/reference/",sidebar:[page("Python API catalog","reference"),group("Model zoo",[
  {
    "label": "Model Zoo",
    "slug": "zoo"
  },
  {
    "label": "ARR-2-EFF-SM",
    "slug": "zoo/arr-2-eff-sm"
  },
  {
    "label": "ARR-4-EFF-SM",
    "slug": "zoo/arr-4-eff-sm"
  },
  {
    "label": "BEAT-2-EFF-SM",
    "slug": "zoo/beat-2-eff-sm"
  },
  {
    "label": "BEAT-3-EFF-SM",
    "slug": "zoo/beat-3-eff-sm"
  },
  {
    "label": "DEN-TCN-LG",
    "slug": "zoo/den-tcn-lg"
  },
  {
    "label": "DEN-TCN-SM",
    "slug": "zoo/den-tcn-sm"
  },
  {
    "label": "DEN-PPG-TCN-SM",
    "slug": "zoo/den-ppg-tcn-sm"
  },
  {
    "label": "SEG-2-TCN-SM",
    "slug": "zoo/seg-2-tcn-sm"
  },
  {
    "label": "SEG-4-TCN-SM",
    "slug": "zoo/seg-4-tcn-sm"
  },
  {
    "label": "SEG-4-TCN-LG",
    "slug": "zoo/seg-4-tcn-lg"
  },
  {
    "label": "SEG-PPG-2-TCN-SM",
    "slug": "zoo/seg-ppg-2-tcn-sm"
  }
]),...apiGroups]},
];
const flatten = (items) => items.flatMap((item) => item.items ? flatten(item.items) : [item]);
export const sectionByPath = Object.fromEntries(sections.flatMap(section => section.sidebar===false ? [[section.href,section.href]] : flatten(section.sidebar).map(item=>[item.slug!==undefined?`/heartkit/${item.slug}/`:item.link,section.href])));
