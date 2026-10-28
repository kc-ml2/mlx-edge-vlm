# Caption Studio

A local Apple Silicon image and video captioning app. Models run through [MLX-VLM](https://github.com/Blaizzy/mlx-vlm); model snapshots are downloaded from Hugging Face into `.models/`. Media is processed on the Mac and is not uploaded to a server.

## Setup

Install Python 3.10 or newer for Apple Silicon, then in this directory run:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

On the Mac where this project was created, Python and dependencies are already installed in `.venv`. Start it with `.venv/bin/python app.py`.

Use your installed Python executable in the first command if it has a different version. Open http://127.0.0.1:8765. Set `PORT` to use another port.

Paste a Hugging Face model URL or `owner/model` and choose **Download & load**. The default is `mlx-community/Qwen2-VL-2B-Instruct-4bit`. Downloads can be several gigabytes. For gated models, authenticate with Hugging Face using `hf auth login` and accept the model's terms on its Hugging Face page first. Only MLX-VLM-supported architectures and compatible checkpoints can run; arbitrary Hugging Face models are not guaranteed to work. Available memory and model size determine whether a model fits. Downloading a model does not convert it to MLX format.

Choose an image and generate a caption, or choose a video and start captioning. During playback the app samples frames, compares them to the last captioned frame, and generates a new caption when the image changes enough. Captions are timestamped at the sampled frame; inference speed can lag behind playback, so frames may be skipped. Sensitivity and sample interval are adjustable. Playback and captioning stop when the video is paused. A running inference may finish after pausing.

The application binds only to localhost and has no authentication. Only use models and model repositories you trust: some model loaders can execute code from model repositories. Do not expose this server to a network. The model and Hugging Face caches use disk space until removed manually.
