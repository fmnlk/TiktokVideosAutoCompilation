#!/usr/bin/python3

"""
Render original meme videos (1080x1920, text + synthetic voice) from a JSON file of ideas.
Each idea gives the scenes of the video, the layout and the caption to post with it.
See ideas/example.json for the format.
"""

###############
### Modules ###
###############

import os
import re
import sys
import json
import argparse
import datetime
import urllib.request
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip, AudioFileClip

WIDTH, HEIGHT, FPS = 1080, 1920, 30
SAMPLE_RATE = 24000
#silence after each scene so the viewer can read
SCENE_PAUSE = 0.35
END_HOLD = 0.9

MODEL_DIR = os.environ.get('KOKORO_DIR', 'models')
MODEL_URL = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'
MODEL_FILES = ['kokoro-v1.0.onnx', 'voices-v1.0.bin']

FONT_BOLD = os.environ.get('MEME_FONT_BOLD', '/usr/share/fonts/opentype/inter/Inter-Bold.otf')
FONT_REGULAR = os.environ.get('MEME_FONT', '/usr/share/fonts/opentype/inter/Inter-Medium.otf')

#(top color, bottom color) of the animated background
PALETTES = [
    ((255, 94, 98), (255, 195, 113)),
    ((67, 56, 202), (236, 72, 153)),
    ((17, 153, 142), (56, 239, 125)),
    ((15, 32, 39), (44, 83, 100)),
    ((252, 70, 107), (63, 94, 251)),
    ((247, 151, 30), (255, 210, 0)),
]

#no color emoji font is available for rendering, emojis are kept for the caption only
EMOJI = re.compile('[\U0001F000-\U0001FAFF☀-➿️‍]')

def clean(text):
    return re.sub(r'\s+', ' ', EMOJI.sub('', text)).strip()

#############
### Voice ###
#############

def downloadModels():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for file in MODEL_FILES:
        path = os.path.join(MODEL_DIR, file)
        if not os.path.exists(path):
            print("Downloading", file, "...")
            urllib.request.urlretrieve(MODEL_URL + file, path)

_kokoro = None

def speak(text, voice, speed):
    """
    Synthesize the text with Kokoro
    OUTPUT: numpy array of samples at SAMPLE_RATE
    """
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        downloadModels()
        _kokoro = Kokoro(*[os.path.join(MODEL_DIR, file) for file in MODEL_FILES])
    text = clean(text)
    if not text:
        return np.zeros(0, dtype=np.float32)
    samples, sr = _kokoro.create(text, voice=voice, speed=speed, lang='en-gb' if voice.startswith('b') else 'en-us')
    assert sr == SAMPLE_RATE
    return samples.astype(np.float32)

################
### Graphics ###
################

def wrap(draw, text, font, maxWidth):
    lines = []
    for paragraph in text.split('\n'):
        line = ''
        for word in paragraph.split():
            candidate = (line + ' ' + word).strip()
            if draw.textlength(candidate, font=font) <= maxWidth or not line:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return '\n'.join(lines)

def textImage(text, fontPath, size, maxWidth, maxHeight=None, fill=(255, 255, 255), stroke=0, strokeFill=(0, 0, 0), align='center'):
    """
    Render a wrapped text as an RGBA PIL image, shrinking the font until it fits maxHeight
    """
    probe = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    while True:
        font = ImageFont.truetype(fontPath, size)
        wrapped = wrap(probe, text, font, maxWidth)
        box = [int(round(v)) for v in probe.multiline_textbbox((0, 0), wrapped, font=font, stroke_width=stroke, align=align, spacing=size // 4)]
        if maxHeight is None or box[3] - box[1] <= maxHeight or size <= 30:
            break
        size -= 6
    image = Image.new('RGBA', (box[2] - box[0] + 4, box[3] - box[1] + 4), (0, 0, 0, 0))
    ImageDraw.Draw(image).multiline_text((2 - box[0], 2 - box[1]), wrapped, font=font, fill=fill, stroke_width=stroke,
                                         stroke_fill=strokeFill, align=align, spacing=size // 4)
    return image

def toArray(image):
    array = np.asarray(image, dtype=np.float32)
    return array[..., :3], array[..., 3:] / 255

def paste(frame, rgb, alpha, x, y, opacity=1.0):
    """
    Alpha blend an image into the frame at x, y (clipped to the frame)
    """
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, WIDTH), min(y + h, HEIGHT)
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x] * opacity
    region = frame[y0:y1, x0:x1]
    region[:] = region * (1 - a) + rgb[y0 - y:y1 - y, x0 - x:x1 - x] * a

def background(palette):
    """
    Return a function giving the animated gradient background at time t
    """
    top, bottom = np.array(palette[0], dtype=np.float32), np.array(palette[1], dtype=np.float32)
    y = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None]

    def frame(t):
        #the gradient slowly breathes up and down
        shift = 0.15 * np.sin(t * 0.8)
        mix = np.clip(y + shift, 0, 1)
        column = top * (1 - mix) + bottom * mix
        return np.repeat(column[:, None, :], WIDTH, axis=1)
    return frame

def easeOut(progress):
    progress = min(max(progress, 0.0), 1.0)
    return 1 - (1 - progress) ** 3

###############
### Layouts ###
###############

def captionLayout(idea, scenes, timings, handle):
    """
    Big centered text per scene (POV, memes, lists) with a pop animation
    """
    bg = background(PALETTES[idea.get('palette', 0) % len(PALETTES)])
    images = []
    for i, scene in enumerate(scenes):
        size = 104 if i == 0 else 92
        image = textImage(clean(scene['text']), FONT_BOLD, size, 920, 1150, stroke=7)
        #pre-rendered scales for the pop animation
        images.append([toArray(image.resize((max(1, int(image.width * s)), max(1, int(image.height * s))), Image.LANCZOS))
                       for s in (0.82, 0.88, 0.94, 0.98, 1.0)])
    handleImage = toArray(textImage(handle, FONT_REGULAR, 40, 900, fill=(255, 255, 255, 200), stroke=2)) if handle else None
    total = timings[-1][1]

    def frame(t):
        image = bg(t)
        index = max(i for i, (start, _) in enumerate(timings) if start <= t or i == 0)
        start = timings[index][0]
        step = min(int((t - start) / 0.03), 4)
        rgb, alpha = images[index][step]
        paste(image, rgb, alpha, (WIDTH - rgb.shape[1]) // 2, (HEIGHT - rgb.shape[0]) // 2 - 60)
        #progress bar
        image[40:52, 60:60 + int((WIDTH - 120) * min(t / total, 1))] = 255
        if handleImage:
            paste(image, *handleImage, (WIDTH - handleImage[0].shape[1]) // 2, HEIGHT - 260)
        return image.astype(np.uint8)
    return frame

def bubbleImage(text, mine):
    textPart = textImage(clean(text), FONT_REGULAR, 58, 720, fill=(255, 255, 255) if mine else (20, 20, 20), align='left')
    padX, padY = 38, 26
    bubble = Image.new('RGBA', (textPart.width + 2 * padX, textPart.height + 2 * padY), (0, 0, 0, 0))
    ImageDraw.Draw(bubble).rounded_rectangle((0, 0, bubble.width - 1, bubble.height - 1), radius=44,
                                             fill=(10, 132, 255) if mine else (233, 233, 235))
    bubble.alpha_composite(textPart, (padX, padY))
    return bubble

def chatLayout(idea, scenes, timings, handle):
    """
    Text conversation where a new bubble appears with each scene
    """
    me = idea.get('me', 'me')
    contact = clean(idea.get('contact', 'Mom'))
    header = Image.new('RGBA', (WIDTH, 230), (246, 246, 246, 255))
    draw = ImageDraw.Draw(header)
    draw.ellipse((WIDTH // 2 - 50, 30, WIDTH // 2 + 50, 130), fill=(160, 160, 170))
    initial = textImage(contact[:1].upper() or '?', FONT_BOLD, 56, 100)
    header.alpha_composite(initial, (WIDTH // 2 - initial.width // 2, 80 - initial.height // 2))
    name = textImage(contact, FONT_REGULAR, 40, 900, fill=(20, 20, 20))
    header.alpha_composite(name, (WIDTH // 2 - name.width // 2, 150))
    draw.line((0, 229, WIDTH, 229), fill=(210, 210, 210), width=2)
    headerArray = toArray(header)

    bubbles = []
    y = 270
    for scene in scenes:
        mine = scene.get('speaker', me) == me
        rgb, alpha = toArray(bubbleImage(scene['text'], mine))
        x = WIDTH - 50 - rgb.shape[1] if mine else 50
        bubbles.append((rgb, alpha, x, y))
        y += rgb.shape[0] + 22
    handleImage = toArray(textImage(handle, FONT_REGULAR, 38, 900, fill=(120, 120, 120))) if handle else None
    white = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.float32)

    def frame(t):
        image = white.copy()
        visible = [i for i, (start, _) in enumerate(timings) if start <= t] or [0]
        last = visible[-1]
        #scroll up so that the last bubble stays above the bottom margin
        _, lastAlpha, _, lastY = bubbles[last]
        target = max(0, lastY + lastAlpha.shape[0] - (HEIGHT - 300))
        appear = easeOut((t - timings[last][0]) / 0.18)
        previousTarget = 0
        if last > 0:
            _, prevAlpha, _, prevY = bubbles[last - 1]
            previousTarget = max(0, prevY + prevAlpha.shape[0] - (HEIGHT - 300))
        scroll = previousTarget + (target - previousTarget) * appear
        for i in visible:
            rgb, alpha, x, y = bubbles[i]
            offset = int((1 - appear) * 40) if i == last else 0
            paste(image, rgb, alpha, x, int(y - scroll) + offset, appear if i == last else 1.0)
        paste(image, *headerArray, 0, 0)
        if handleImage:
            paste(image, *handleImage, (WIDTH - handleImage[0].shape[1]) // 2, HEIGHT - 150)
        return image.astype(np.uint8)
    return frame

LAYOUTS = {'caption': captionLayout, 'chat': chatLayout}

##############
### Render ###
##############

def slugify(text):
    return re.sub(r'[^a-z0-9]+', '-', clean(text).lower()).strip('-')[:50] or 'video'

def render(idea, outDir, handle=''):
    """
    Render one idea into outDir/<slug>.mp4 and write the caption to post in outDir/<slug>.txt
    """
    scenes = idea['scenes']
    voices = idea.get('voices', {})
    defaultVoice = idea.get('voice', 'am_adam')
    speed = idea.get('speed', 1.1)
    #voice over of every scene, the scene lasts as long as its voice
    audio = []
    timings = []
    t = 0.0
    for scene in scenes:
        voice = scene.get('voice') or voices.get(scene.get('speaker', ''), defaultVoice)
        samples = speak(scene.get('say', scene['text']), voice, speed)
        duration = max(len(samples) / SAMPLE_RATE, scene.get('min_duration', 1.0)) + SCENE_PAUSE
        padded = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        padded[:len(samples)] = samples
        audio.append(padded)
        timings.append((t, t + duration))
        t += duration
    audio.append(np.zeros(int(END_HOLD * SAMPLE_RATE), dtype=np.float32))
    timings[-1] = (timings[-1][0], t + END_HOLD)
    total = t + END_HOLD

    os.makedirs(outDir, exist_ok=True)
    slug = slugify(idea.get('title', scenes[0]['text']))
    wavPath = os.path.join(outDir, slug + '.wav')
    sf.write(wavPath, np.concatenate(audio), SAMPLE_RATE)

    videoPath = os.path.join(outDir, slug + '.mp4')
    try:
        layout = LAYOUTS[idea.get('layout', 'caption')]
        clip = VideoClip(layout(idea, scenes, timings, handle), duration=total)
        voiceOver = AudioFileClip(wavPath)
        clip = clip.with_audio(voiceOver)
        clip.write_videofile(videoPath, fps=FPS, codec='libx264', audio_codec='aac', preset='medium',
                             ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '20'], logger=None)
        voiceOver.close()
    finally:
        os.remove(wavPath)

    hashtags = ' '.join('#' + tag.lstrip('#') for tag in idea.get('hashtags', []))
    with open(os.path.join(outDir, slug + '.txt'), 'w') as f:
        f.write(idea.get('caption', '') + '\n\n' + hashtags + '\n')
        if idea.get('sound'):
            f.write('\nSound: ' + idea['sound'] + '\n')
        if idea.get('why'):
            f.write('\nWhy it can work: ' + idea['why'] + '\n')
    print("Video saved in", videoPath, "(%.1fs)" % total)
    return videoPath

def main():
    parser = argparse.ArgumentParser(description="Render original meme videos from a JSON file of ideas")
    parser.add_argument('ideas', help="JSON file with a list of ideas (see ideas/example.json)")
    parser.add_argument('--out', help="output folder (default: output/<today>)")
    parser.add_argument('--handle', default='', help="text shown at the bottom of the video, ex: @myaccount")
    parser.add_argument('--only', type=int, help="render only the idea at this index")
    args = parser.parse_args()

    with open(args.ideas) as f:
        ideas = json.load(f)
    outDir = args.out or os.path.join('output', datetime.date.today().isoformat())
    selected = [ideas[args.only]] if args.only is not None else ideas
    failed = 0
    for idea in selected:
        try:
            render(idea, outDir, args.handle)
        except Exception as e:
            failed += 1
            print("Error rendering", idea.get('title'), ":", e, file=sys.stderr)
    sys.exit(1 if failed else 0)

if __name__ == '__main__':
    main()
