# Arrhythmia Demo

A demo is provided to showcase the capabilities of the arrhythmia classifier models. Similar to other modes, the demo can be invoked either via CLI or within `heartkit` python package. At a high level, the demo performs the following actions based on the provided configuration parameters:

1. Load the configuration file (e.g. `demo-arrhythmia.json`)
1. Load the desired dataset features (e.g. `icentia11k`)
1. Load the trained model (e.g. `arrhythmia-2-class`)
1. Load random test subject's data
1. Perform inference either on PC or EVB
1. Generate report

---

## <span class="sk-h2-span">Usage</span>

### PC backend

1. Create / modify configuration file (e.g. `demo-arrhythmia.json`)
1. Ensure "pc" is selected as the backend in configuration file.
1. Run demo `heartkit --mode demo --task arrhythmia --config ./configs/demo-arrhythmia.json`
1. HTML report will be saved to `${job_dir}/report.html`

### EVB backend

1. Create / modify configuration file (e.g. `demo-arrhythmia.json`)
1. Ensure "evb" is selected as the backend in configuration file.
1. Plug EVB into PC via two USB-C cables.
1. Build and flash firmware to EVB `cd evb && make && make deploy`
1. Run demo `heartkit --mode demo --task arrhythmia --config ./configs/demo-arrhythmia.json`
1. HTML report will be saved to `${job_dir}/report.html`

---

## <span class="sk-h2-span">Outputs</span>


<div class="sk-plotly-graph-div">
--8<-- "assets/arrhythmia-demo.html"
</div>

---
