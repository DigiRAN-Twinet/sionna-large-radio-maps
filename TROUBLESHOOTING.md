Troubleshooting / FAQ
=====================

## My GPU is running out of memory when computing radio maps

The radio map computation script tries to automatically split the work into chunks that fit into your GPU memory. Estimating the exact amount of memory that can be allocated for the radio map computation is not straightforward, therefore the script uses a safety buffer of 3GB to avoid running out. If your scene is very complex, this assumption may be wrong, causing the script to overestimate the amount of available memory and therefore run out of VRAM.

If that happens, you can manually tweak the preconfigured memory size by setting the environment variable `ASSUMED_SCENE_SIZE_MIB` to the desired buffer size in MiB.

## Can I simulate diffuse reflections ?

Sionna RT supports simulating diffuse reflections, but this repository only enables specular paths by default since they are the ones carying most of the energy. It can be enabled by modifying the call to the radio map solver in [sionna_lrm/radio_maps.py](sionna_lrm/radio_maps.py#L208-215) and adding the flag `diffuse_reflection=True`.
