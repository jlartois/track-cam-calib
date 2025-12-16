# Camera calibration through Optical-Flow Keypoint Tracking

Estimate a moving camera's pose through optical-flow keypoint tracking.

To calibrate `video.mkv` taken by a pinhole camera:

```bash
python track_pinhole.py
```

To calibrate a 360° camera video `video360.mp4` ([download here](https://cloud.ilabt.imec.be/index.php/s/i3tGBSNHE3WQfez)), first reproject it to a cubemap (only keeping the front face):

```bash
python reproject.py
```

This will result in `video360_repr.mp4`. For simplicity, we provide`video360_repr.mp4` directly in this repository. Then run:

```bash
python track_360.py
```

Both files allow to set parameters, rather than expecting command line arguments.

The GIF below shows the keypoints being tracked over time. (Note that due to compression, the GIF has a much lower quality than the original video.)

![Example](example.gif)

# Notes
This repository is a simple proof of concept. It used Blender to render out a posetrace, and store each pose's extrinsics in `camera_poses.json`. That way, the estimated extrinsics can be evaluated against the groundtruth.
This means that the Blender axial system is used. For actual applications, COLMAP would be used to estimate the intrinsics and extrinsics of a subset of the frames (for example every 25th frame), and the remaining frames' extrinsics would be estimated using the code in this repository. Therefore, the code should be adapted to handle the COLMAP axial system.

Additionally, we assume that the video is already undistorted.

Dataset and paper by [IDLab MEDIA](https://media.idlab.ugent.be/).
