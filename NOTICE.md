# Notice

streamdetection is based on clearcam (https://github.com/roryclear/clearcam), copyright Rory Clear, licensed under the GNU General Public License v3.0. This modified version is distributed under the same license.

Based on clearcam commit `cf3909f8a622408be3b5f9811e4a9a56426b3134` (2026-10-28).

## Changes made in streamdetection (2026-10-30, Koen Korevaar)
- New live dashboard (`dashboard.html`) served at `/`; the original interface moved to `/classic`
- Stream links from YouTube, Twitch and similar sites resolved with yt-dlp (`utils/helpers.py`)
- JSON and image endpoints for the dashboard: `/api/streams`, `/api/frame`, `/api/events`, `/api/event_thumb`
- Live detection state and first-sighting events with thumbnails kept per stream (`clearcam.py`)
- One ffmpeg process per stream records HLS and pipes frames to the detector, capped at 20 fps
- Local webcam sources (`webcam:<device>`) and a `--webcam` flag
- Windows support: `-fps_mode` instead of the removed `-vsync`, forward-slash paths for ffmpeg, a tinygrad import shim (`utils/windows_compat.py`), and a main loop that no longer starves the frame reader thread
- Model downloads use the operating system certificate store (truststore)
