# SubPass Sample

For mobile tile-based GPUs, subpasses is one of the most important ways to save memory bandwidth, therefore, to improve power efficiency and help performance at the meantime. 

SubPass sample demonstrate the use of vulkan subpasses to perform a filmic tonemapping operator (on a simple forward rendered scene) and the impact on bandwidth and performance with subpass.

Optionally runs the tonemap as a subpass of the main scene pass.  Has onscreen UI controls to  enable/disable the use of subpasses (for measuring GPU subpass efficiency).

Bandwidth savings are meaused with Snapdragon Profiler. 60% of the bandwidth can be saved in this sample when enabling subpass.