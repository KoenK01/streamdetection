# streamdetection

Live object detection on any video stream. Paste a YouTube live link, a Twitch channel, an RTSP camera, an HLS `.m3u8` feed or a local webcam, and a dashboard shows every stream with live bounding boxes, an activity feed of new objects and per-class totals.

Built on [clearcam](https://github.com/roryclear/clearcam) by Rory Clear. See [Credits](#credits).

## Features
- Dashboard with a live tile per stream: the exact frame the detector processed, with its boxes, so boxes never lag the picture
- YouTube and Twitch links resolved to their HLS feed with yt-dlp, re-resolved on every reconnect because those links expire
- Pedestrians and cyclists on high, overhead cameras through a motion layer: small movers the detector can't recognise are tracked and labelled by real-world speed (walking under ~2.6 m/s, cycling under ~9 m/s), with the scale learned from the cars on that camera
- People on bikes are shown as one cyclist / motorcyclist box
- Activity feed with a crop of each new object, session totals per class
- Class filter, minimum confidence slider, focus mode (double click a tile)
- Recordings kept as HLS per stream per day
- Runs on Windows, Linux and macOS; GPU inference through tinygrad (OpenCL on AMD and others)

## Requirements
- Python 3.11 or later
- ffmpeg 5.1 or later on your PATH
- Node.js (optional, recommended): some YouTube streams only resolve when yt-dlp can solve a JS challenge with it

## Run
```
pip install -r requirements.txt
python clearcam.py
```
Open http://localhost:8080 and click **Add stream**. The first frame takes a while on the very first run while the GPU kernels compile; after that they are cached.

- `python clearcam.py --webcam` adds the first local webcam (Windows), or pick one with `--webcam="c922 Pro Stream Webcam@1920x1080@30"`
- webcam sources in the dashboard use `webcam:<device>[@WIDTHxHEIGHT][@FPS]`
- `DEV=AMD` / `DEV=NV` picks a GPU backend if tinygrad falls back to the CPU
- the original clearcam interface is still available at http://localhost:8080/classic

## Credits
This project is a modified version of **[clearcam](https://github.com/roryclear/clearcam)** by Rory Clear, licensed under the GNU GPL v3. The detection pipeline (YOLOv9 on tinygrad), OC-SORT tracking, recording, alerts and the classic interface all come from clearcam. The original README is kept in [CLEARCAM_README.md](CLEARCAM_README.md), and [NOTICE.md](NOTICE.md) lists what changed here.

Also built with [tinygrad](https://github.com/tinygrad/tinygrad), [OC-SORT](https://github.com/noahcao/OC_SORT), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [ffmpeg](https://ffmpeg.org) and [OpenCV](https://opencv.org).

Notifications, remote viewing and the other premium features inherited from clearcam talk to clearcam.org and need a clearcam account; they are not part of this project.

## License
GNU General Public License v3.0, the same license as clearcam. See [LICENSE.md](LICENSE.md).
