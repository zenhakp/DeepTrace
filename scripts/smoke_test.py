"""CPU smoke test: environment + EfficientNet-B4 forward pass.
This checks that the software runs. It is NOT a research result."""
import platform
import time

import timm
import torch

from deeptrace.utils.runtime import resolve_device, set_seed


def main() -> None:
    set_seed(42)
    device = resolve_device("auto")
    print(f"python  : {platform.python_version()}")
    print(f"torch   : {torch.__version__}")
    print(f"timm    : {timm.__version__}")
    print(f"device  : {device}")

    # pretrained=False: no download needed for a pure software check
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params  : {n_params:,}")

    x = torch.randn(2, 3, 380, 380, device=device)  # 380 = EfficientNet-B4 native resolution
    start = time.time()
    with torch.no_grad():
        logits = model(x)
    elapsed = time.time() - start

    assert logits.shape == (2, 2), f"Unexpected logits shape: {tuple(logits.shape)}"
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2, device=device), atol=1e-5)
    print(f"logits  : shape {tuple(logits.shape)}, forward {elapsed:.2f}s")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
