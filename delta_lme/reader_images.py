"""Provider image limits applied in wire order, without changing text evidence."""
from .checkpoint import fingerprint


def image_limit(model):
    return 10 if model == "Qwen/Qwen3.5-9B" else None


def limit_images(messages, limit):
    if limit is not None and limit < 0:
        raise ValueError("image limit must be nonnegative")
    images = [(mi, pi, part) for mi, message in enumerate(messages)
              if isinstance(message.get("content"), list)
              for pi, part in enumerate(message["content"])
              if part.get("type") in ("image_url", "input_image")]
    kept = images if limit is None else images[:limit]
    removed = {(mi, pi) for mi, pi, _ in images[len(kept):]}
    packet = [{**message, "content": [part for pi, part in enumerate(message["content"])
                                     if (mi, pi) not in removed]}
              if isinstance(message.get("content"), list) else dict(message)
              for mi, message in enumerate(messages)]
    audit = {"policy": "first images in request order; retain all text", "limit": limit,
             "original_images": len(images), "sent_images": len(kept),
             "images": [{"message_index": mi, "part_index": pi,
                         "sha256": fingerprint(part), "sent": (mi, pi) not in removed}
                        for mi, pi, part in images]}
    return packet, audit
