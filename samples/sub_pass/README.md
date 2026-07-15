# Subpass Sample

For mobile tile-based GPUs, subpasses is one of the most important ways to save memory bandwidth, therefore, to improve power efficiency and help performance at the meantime. 

Subpass sample demos the use of vulkan subpasses to perform a filmic tonemapping operator (on a simple forward rendered scene) and the impact on bandwidth and performance with subpass.
Optionally runs the tonemap as a subpass of the main scene pass.  Has onscreen UI controls to  enable/disable the use of subpasses (for measuring GPU subpass efficiency).
Bandwidth savings are meaused with Snapdragon Profiler. 60% of the bandwidth can be saved in this sample when enabling subpass.

From SnapdragonProfiler, we can see that there are four surfaces when subpass is disabled and three surfaces otherwise. There is extra GMEM stores when there is no subpass. 

![NoSubPassSurfaces](img/nosubpassstage.PNG)

![SubPassSurfaces](img/subpassstage.PNG)

Read total and write total have both been reduced when subpass is enable.

![NoSubPassBandwidth](img/nosubpass.PNG)

![SubPassBandwidth](img/subpass.PNG)

## Running

- If you haven't already, setup the framework and build the code [instructions here](../../README.md#configuring)
- Running this sample has no special additional requirements [instructions here](../../README.md#running)
