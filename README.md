# Tiktok  Videos AutoCompilation Project

Everyday, we ( made an automatique compiled video of the best Tiktok from the trending list and other categories. 
We select those videos according to a "ranking algo" (to be improve). You can see all videos on  [Youtube](https://www.youtube.com/channel/UCpbFh9sU36HmfcaCgTJeSgg)  

## Built With

* [TikTokApi](https://github.com/davidteather/TikTok-Api) - Used to retrieve trending and music challenge videos (Playwright based)
* [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Used to download videos and refresh their stats
* [MoviePy](https://zulko.github.io/moviepy/) - Used to merge and render videos (needs ffmpeg)

## Setup

```
pip install -r requirements.txt
python -m playwright install chromium
```

TikTok blocks most bots. To reduce it, set the `MS_TOKEN` environment variable with the value of the `msToken` cookie of a tiktok.com session opened in your browser.

## Usage

```
python videomaker.py import-trending --count 300      # add the current trending videos to dataVideo.txt
python videomaker.py import-music --count 100         # add the videos of the musics of listChallenge.txt to dataVideoChallenge.txt
python videomaker.py update-db [--prune]              # refresh the stats of the videos of the DB
python videomaker.py filter                           # remove the videos that are not popular enough
python videomaker.py make-video --period week --period-number 13 --year 2020 --ranking trending --number 50
```

`make-video` downloads the top videos of the period in a folder, merges them in `<content>_<year>_<period>_<number>.mp4` and marks them as used in the DB (use `--skip-used` to ignore them next time).
...

## Original meme videos

`memevideo.py` renders original, ready-to-post videos (1080x1920, animated text + synthetic voice with [Kokoro](https://github.com/thewh1teagle/kokoro-onnx)) from a JSON file of ideas. Two layouts:

* `caption` - big centered text per scene (POV, memes, lists)
* `chat` - a text conversation where a bubble appears with each line
* `story` - a narrated 60-90 s story: title card for the hook, then word by word captions on an animated background (see `ideas/story-example.json`)

```
python memevideo.py ideas/example.json --handle @myaccount   # videos in output/<today>/
```

Each video comes with a `.txt` containing the caption, hashtags and a sound suggestion. The voice model (~350 MB) is downloaded into `models/` on first run. Add a trending sound at low volume in the TikTok app and enable the "AI-generated content" label since the voice is synthetic.
