# Camera calibration through Optical-Flow Keypoint Tracking

Estimate a moving camera's pose through optical-flow keypoint tracking.

To calibrate `video.mkv` taken by a pinhole camera:

```bash
python track_pinhole.py
```

To calibrate `video360.mp4` a 360° camera video, first reproject it to a cubemap (only keeping the front face):

```bash
python reproject.py
```

This will result in `video360_repr.mp4`. The original `video360.mp4` is too large for Github, so we provide`video360_repr.mp4` directly. Then run:

```bash
python track_360.py
```

Both files allow to set parameters, rather than expecting command line arguments.

The GIF below shows the keypoints being tracked over time. (Note that due to compression, the GIF has a much lower quality than the original video.)

![Example](example.gif)
