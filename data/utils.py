import numpy as np
import random


def parse_event(event):

    event = np.array(event)            
    event = np.vstack([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T      # (n. 4)

    event = event.astype(np.float32)

    # Account for int-type timestamp
    event[:, 2] /= 1e6                              # e.g. us --> s

    # Account for zero polarity
    if event[:, 3].min() >= -0.5:
        event[:, 3][event[:, 3] <= 0.5] = -1        # e.g. 0.0 --> -1.0

    return event

def split_event(event, length):
    # Randomly select a time window
    if len(event) > length:
        start = random.choice(range(len(event) - length + 1))
        event = event[start: start + length]

    return event

def make_event_histogram(event, resolution, red, blue, background_mask=True, count_non_zero=False, thresh=10.):
    # count the number of positive and negative events per pixel
    H, W = resolution
    pos_x, pos_y = event[:, 0][event[:, 3] > 0].astype(np.int32), event[:, 1][event[:, 3] > 0].astype(np.int32)
    pos_count = np.bincount(pos_x + pos_y * W, minlength=H * W).reshape(H, W)
    neg_x, neg_y = event[:, 0][event[:, 3] < 0].astype(np.int32), event[:, 1][event[:, 3] < 0].astype(np.int32)
    neg_count = np.bincount(neg_x + neg_y * W, minlength=H * W).reshape(H, W)
    hist = np.stack([pos_count, neg_count], axis=-1)  # [H, W, 2]

    # remove hotpixels, i.e. pixels with event num > thresh * std + mean
    if thresh > 0:
        if count_non_zero:
            mean = hist[hist > 0].mean()
            std = hist[hist > 0].std()
        else:
            mean = hist.mean()
            std = hist.std()
        hist[hist > thresh * std + mean] = 0

    # normalize
    hist = hist.astype(np.float32) / hist.max()  # [H, W, 2]

    # colorize
    cmap = np.stack([red, blue], axis=0).astype(np.float32)  # [2, 3]
    img = hist @ cmap  # [H, W, 3]

    # alpha-masking with pure white background
    if background_mask:
        weights = np.clip(hist.sum(-1, keepdims=True), a_min=0, a_max=1)
        background = np.ones_like(img) * 255.
        img = img * weights + background * (1. - weights)

    img = np.round(img).astype(np.uint8)  # [H, W, 3], np.uint8 in (0, 255)

    return img

def center_event(events, resolution):
    # Center the temporal & spatial coordinates of events.
    
    # temporal
    events[:, 2] -= events[:, 2].min()
    # spatial
    H, W = resolution
    x_min, x_max = events[:, 0].min(), events[:, 0].max()
    y_min, y_max = events[:, 1].min(), events[:, 1].max()
    x_shift = ((x_max + x_min + 1.) - W) // 2.
    y_shift = ((y_max + y_min + 1.) - H) // 2.
    events[:, 0] -= x_shift
    events[:, 1] -= y_shift
    return events

def _convert_image_to_rgb(image):
    return image.convert("RGB")

# -------------------------------------------------------event_augment_start------------------------------------------------

def event_augment(event, cfg, resolution):
        augment_random_time_flip = getattr(cfg, 'augment_random_time_flip', False)
        augment_random_flip_events_along_x = getattr(cfg, 'augment_random_flip_events_along_x', False)
        augment_random_shift_events = getattr(cfg, 'augment_random_shift_events', False)

        if augment_random_time_flip:
            event = random_time_flip(event)
        if augment_random_flip_events_along_x:
            event = random_flip_events_along_x(event, resolution)
        if augment_random_shift_events:
            event = random_shift_events(event, resolution)

        return event

# 上下左右随机平移
def random_shift_events(events, resolution, max_shift=20):
    """Spatially shift events by a random offset."""
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2, ))
    events[:, 0] += x_shift
    events[:, 1] += y_shift

    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]
    return events

# 水平翻转
def random_flip_events_along_x(events, resolution, p=0.5):
    """Flip events along horizontally with probability p."""
    H, W = resolution
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]
    return events

# 极性翻转
def random_time_flip(events, p=0.5):
    """Flip events over time with probability p."""
    if np.random.random() < p:
        events = np.flip(events, axis=0)
        events = np.ascontiguousarray(events)
        # reverse the time
        events[:, 2] = events[0, 2] - events[:, 2]
        # reverse the polarity
        events[:, 3] = -events[:, 3]
    return events

# -------------------------------------------------------event_augment_end------------------------------------------------

