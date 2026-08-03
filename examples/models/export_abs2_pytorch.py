from pathlib import Path

import torch


class Abs2(torch.nn.Module):
    def forward(self, value):
        return torch.abs(value)


def main():
    model = Abs2().eval()
    example_input = torch.zeros(2, dtype=torch.float32)
    exported = torch.export.export(model, (example_input,))
    torch.export.save(exported, Path(__file__).with_name("abs2_pytorch.pt2"))


if __name__ == "__main__":
    main()
