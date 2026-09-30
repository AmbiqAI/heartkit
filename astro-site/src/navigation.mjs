import sidebar from "./data/sidebar.json" with { type: "json" };
import apiSidebar from "./data/api-sidebar.json" with { type: "json" };
const page = (label, slug) => ({ label, slug });
const group = (label, items) => ({ label, items, collapsed: true });
const original = (label) => sidebar.find((item) => item.label === label).items;
const apiGroups = Object.entries(Object.groupBy(apiSidebar, (item) => item.label.split(".")[1] || "Package")).map(([label, items]) => ({label, items, collapsed:true}));
export const sections = [
 {label:"Home",href:"/heartkit/",sidebar:false},
 {label:"Getting started",href:"/heartkit/quickstart/",sidebar:[page("Install heartKIT","quickstart"),page("Use the command line","usage/cli"),page("Use Python","usage/python")]},
 {label:"User guide",href:"/heartkit/guides/",sidebar:[page("Guides and examples","guides"),group("Model workflow",[page("Workflow overview","modes"),page("Configure a task","modes/configuration"),page("Download datasets","modes/download"),page("Train a model","modes/train"),page("Evaluate a model","modes/evaluate"),page("Export a model","modes/export"),page("Run a demo","modes/demo")]),group("Datasets",original("Datasets")),group("Models",original("Models")),group("Examples",[
 page("Custom task","guides/byot"),page("Arrhythmia training","guides/train-arrhythmia-model"),page("ECG denoising","guides/train-ecg-denoiser"),page("ECG segmentation","guides/train-ecg-segmentation"),page("ECG foundation model","guides/ecg-foundation-model")])]},
 {label:"Tasks",href:"/heartkit/tasks/",sidebar:original("Tasks")},
 {label:"Reference",href:"/heartkit/reference/",sidebar:[page("Python API catalog","reference"),group("Model zoo",original("Model Zoo")),...apiGroups]},
];
const flatten = (items) => items.flatMap((item) => item.items ? flatten(item.items) : [item]);
export const sectionByPath = Object.fromEntries(sections.flatMap(section => section.sidebar===false ? [[section.href,section.href]] : flatten(section.sidebar).map(item=>[item.slug!==undefined?`/heartkit/${item.slug}/`:item.link,section.href])));
