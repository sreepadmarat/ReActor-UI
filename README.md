# ReActor UI Modernized
### Fast, high-resolution face swapping application with multi-provider hardware acceleration

> Extended and modernized fork of ReActor / roop with enhanced performance, modernized UI, multi-provider hardware acceleration, and dynamic high-res face detection.

Take a video or photo and replace the face in it with a face of your choice. You only need one image of the desired face — no dataset or training required.

![demo-gif](docs/demo.gif)

---

## 🌟 Key Features & Improvements
* **Multi-Provider Hardware Acceleration**: Dynamic support for `CoreML` (macOS Apple Silicon), `DirectML` (AMD / Direct3D), `CUDA` / `TensorRT` (NVIDIA), `ROCm`, and `CPU`.
* **Zero-Disk I/O In-Memory Pipe Streaming**: Option for memory-buffered video frame streams avoiding heavy temporary PNG disk writes.
* **Modernized UI (CustomTkinter)**: Redesigned window layout (850x850), modernized toggle switches, and distinct action buttons.
* **High-Resolution Model Fallbacks**: Dynamic model selection supporting $512\times 512$ (`simswap_512.onnx`), $256\times 256$ (`blendswap_256.onnx`), and $128\times 128$ (`inswapper_128.onnx`).
* **Face Restoration & Enhancers**: Supports both `GFPGAN` and `CodeFormer` face enhancement models.
* **Dynamic High-Res Face Detection**: Automatically scales `det_size` up to `(1280, 1280)` on HD/4K videos for accurate distant face tracking.

---

## 🛠 How do I install it?

### Prerequisites
Make sure you have `ffmpeg` and `python3.9`+ (up to Python 3.14) installed on your system.

### Quick Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sreepadmarat/ReActor-UI.git
   cd ReActor-UI
   ```

2. **Create and Activate Virtual Environment**
   - **Bash / Zsh**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Fish Shell**:
     ```fish
     python3 -m venv venv
     source venv/bin/activate.fish
     ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python3 run.py
   ```

---

## 🚀 Usage

Executing `python3 run.py` will launch the modernized desktop interface:

1. Select a **Source Face** (image containing the face you want to use).
2. Select a **Target File** (image or video in which you want to replace the face).
3. Toggle options like **Face Enhancer**, **Keep audio**, or **Keep fps**.
4. Click **Start** to process or **Preview** to check single-frame alignment.

### CLI Mode Options
You can also run ReActor in headless CLI mode:
```bash
python3 run.py -s source.jpg -t target.mp4 -o output.mp4 --execution-provider directml cpu
```

```
options:
  -h, --help            show this help message and exit
  -s SOURCE_PATH, --source SOURCE_PATH
                        select a source image
  -t TARGET_PATH, --target TARGET_PATH
                        select a target image or video
  -o OUTPUT_PATH, --output OUTPUT_PATH
                        select output file or directory
  --frame-processor {face_swapper,face_enhancer} [{face_swapper,face_enhancer} ...]
                        pipeline of frame processors
  --keep-fps            keep original fps
  --keep-audio          keep original audio
  --keep-frames         keep temporary frames
  --many-faces          process every face
  --video-encoder {libx264,libx265,libvpx-vp9}
                        adjust output video encoder
  --video-quality VIDEO_QUALITY
                        adjust output video quality
  --max-memory MAX_MEMORY
                        maximum amount of RAM in GB
  --execution-provider {tensorrt,cuda,coreml,directml,rocm,cpu}
                        execution provider
  --execution-threads EXECUTION_THREADS
                        number of execution threads
  -v, --version         show program's version number and exit
```

---

## 📜 Credits
- [Gourieff](https://github.com/Gourieff): Original ReActor project author
- [s0md3v](https://github.com/s0md3v/roop): Author of the original roop application
- [insightface](https://github.com/deepinsight/insightface): Facial detection and representation models
- [ffmpeg](https://ffmpeg.org/): Multimedia framework
