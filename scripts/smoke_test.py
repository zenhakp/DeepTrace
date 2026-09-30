"""CPU smoke test: environment + EfficientNet-B4 forward pass.
This checks that the software runs. It is NOT a research result."""
import time

import timm
import torch

from deeptrace.common.device import describe_environment, resolve_device
from deeptrace.common.seed import seed_everything


def main() -> None:
    seed_everything(42)
    device = resolve_device("auto")
    for key, value in describe_environment().items():
        print(f"{key:15}: {value}")
    print(f"{'timm':15}: {timm.__version__}")
    print(f"{'device':15}: {device}")

    # pretrained=False: no download needed for a pure software check
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2).to(device).eval()
    print(f"{'params':15}: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 3, 380, 380, device=device)  # 380 = EfficientNet-B4 native resolution
    start = time.time()
    with torch.no_grad():
        logits = model(x)
    elapsed = time.time() - start

    assert logits.shape == (2, 2), f"Unexpected logits shape: {tuple(logits.shape)}"
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2, device=device), atol=1e-5)
    print(f"{'logits':15}: shape {tuple(logits.shape)}, forward {elapsed:.2f}s")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
