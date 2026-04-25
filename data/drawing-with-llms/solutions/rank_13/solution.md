# 13th Place Solution - A Kaggle beginner's attempt

- **Author:** Howard（ホワルド）
- **Date:** 2025-05-28T03:04:30.927Z
- **Topic ID:** 581032
- **URL:** https://www.kaggle.com/competitions/drawing-with-llms/discussion/581032

**GitHub links found:**
- https://github.com/mit-han-lab/nunchaku

---

Congratulations to the winners and all participants for their efforts!

I want to give a huge shoutout to @richolson for his unbelievably educational and helpful notebook and discussion posts. I personally learned a lot from [his notebook](https://www.kaggle.com/code/richolson/stable-diffusion-svg-scoring-metric) and it was the starting point for my solution.

EDIT: I published my version of the [notebook](https://www.kaggle.com/code/howarudo/13th-place-solution-flux-svg-scoring)! Feel free to check it out!

# Summary
- `'a'` for OCR mitigation
- Optimized VTracer's bitmap -> SVG conversion with binary search and SVG compression
- Used FLUX.1 Schnell and [nunchaku quantization](https://github.com/mit-han-lab/nunchaku)
- VQA with dummy question and answer

# OCR mitigation
From previous discussions, we know that adding a letter to the iamge can reduce OCR hallucinations. I added the letter `'a'` to the bottom right corner of the image.

```svg
svg_a = """<path d="M329 341q-1.2 1-2 1.5t-2.4.45q-2 0-3-1t-1-3q0-1 .5-1.7.4-1 1-1.3.7-.45 1.6-.7.7 0 2-.4q2.7-.3 4-.8v-.6q0-1.4-.7-2-.9-.8-2.5-.8t-2.3.6-1 2l-2-.3q.3-1.4 1-2.2.7-.9 2-1.3 1.3-.5 3-.5t2.7.4 1.6 1 .7 1.5q.1.6.1 2v3q0 3 .15 3.9.2.8.6 1.6h-2.3q-.4-.7-.5-1.6m-.2-5q-1 .45-3.6.8-1.4.2-2 .5t-1 .7q-.3.5-.3 1 0 1 .7 1.5.7.6 2 .6t2.3-.6 1.5-1.5q.4-.8.4-2.2z" stroke="#fff"/>"""
```

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17096593%2F6702f89386eee00cc5a472dfcc64fd33%2Fa.png?generation=1748401229132058&alt=media)

# Bitmap to SVG conversion
## Binary Search
In a [public notebook](https://www.kaggle.com/code/dhurky/lb-0-52-sd-svg-conversion-using-vtracer), VTracer was used for SVG conversion but in that pipeline, the bitmap was resized to a constant smaller size to reduce the SVG size (10kB restriction). One improvement to this is to use binary search to find the maximum size to convert the bitmap to SVG without exceeding the 10kB limit.

## SVG Compression
VTracer's output SVG looks like this:

```svg
<path d="M0,0 L1,0 L1,24 L0,57 L6,57 L6,23 L50,23 L50,56 L61,56 L60,47 L60,24 L74,24 L74,36 L88,38 L88,7 L89,7 L89,45 Z " fill="#087905" transform="translate(26,148)"/>
```
Compression techniques:
- **Absolute to Relative Coordinates**: Convert absolute coordinates to relative coordinates in the SVG path.
- **Removing Transformations**: Remove unnecessary transformations like `translate`.
- **Longest path -> fill**: Convert the longest path to a fill rectangle.

```svg
<path d="M26 148h1v24l-1 33h6v-34h44v33h11l-1-9v-23h14v12l14 2v-31h1v38Z" fill="#087905"/>
```
This allows bitmaps to have be resized about 50px bigger than without compression. It does not affect images that are relatively SVG-friendly but it helps with images that are not.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17096593%2Ff70b7938bcbd9296c77a613e80f14565%2Fcompresion_1.png?generation=1748401292468823&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17096593%2Ffec70c2c25ddfa192f4b64dd8473b1fb%2Fcompression_2.png?generation=1748401301655450&alt=media)

# Image Generation
I used [FLUX.1 Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) for image generation and [nunchaku](https://github.com/mit-han-lab/nunchaku) for quantization. I tried base SD models, finetuning SDXL Lora but found that FLUX.1 Schnell has better prompt adherence. Also, with quantization, FLUX.1 Schnell is faster.

# Dummy VQA
From 10-12 images generated, I evaluated the images using the questions below.

```python
question = [
    f'Does this image primarily represent "{description}"?',
    f'Is "{description}" clearly shown in the image??'
]
choices = [
    ['definitely', 'somewhat', 'barely', 'no'],
    ['yes', 'somewhat', 'no']
]
answer = [
    'definitely',
    'yes'
]
```
In addition, I performed a linear regression on fidelity VQA scores using dummy VQA scores from an extended training set to predict the final VQA score.

$$VQA_{predicted} = w_1 * VQA_{dummy_1} + w_2 * VQA_{dummy_2} + b$$

Evaluating an image per question takes `< 2 seconds` while generating an image with FLUX.1 Schnell takes `~ 1 second`. Hence, I found that generating more images and evaluating them with 1-2 questions is more efficient than generating fewer images and evaluating them with 3-4 questions.

# In Hindsight...
After reading @cnumber's post, I regret not trying `pydiffvg` to further optimize the scores and not trying Siglip. I also had moments where I wished I had participated in a team as it was difficult to come up with ideas solo. Regardless, I am happy with the knowledge and experience I gained from this competition!

**PS: In future competitions, I hope to join a team to bounce ideas off each other so feel free to reach out if you happen to see me in a future competition!**
