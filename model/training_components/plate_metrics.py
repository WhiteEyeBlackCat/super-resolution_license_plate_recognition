import re
from collections import Counter


# TO do: 亂碼改"?"
def extract_plate(text, target_len=None):
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9?]", "", text)
    if len(text) == 0:
        return None
    if target_len is not None:
        return text[:target_len] if len(text) >= target_len else text
    return text


def plate_score(pred, gt):
    if pred is None or len(pred) != len(gt):
        return None
    score = 0
    for p, g in zip(pred, gt):
        if p == g:
            score += 2
        elif p == "?":
            continue
        else:
            score -= 1
    return score



# TO do: what if got two same most common?
def vote_plate(preds, target_len=None, ignore_chars={"?"}):
    preds = [p for p in preds if p is not None]
    if target_len is not None:
        preds = [p for p in preds if len(p) == target_len]
    if not preds:
        return None

    plate_len = len(preds[0])
    if any(len(p) != plate_len for p in preds):
        return None

    final_chars = []
    for i in range(plate_len):
        chars = [p[i] for p in preds if p[i] not in ignore_chars]
        if len(chars) == 0:
            final_chars.append("?")
            continue
        counter = Counter(chars)
        final_chars.append(counter.most_common(1)[0][0])
    return "".join(final_chars)