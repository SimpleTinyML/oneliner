from pathlib import Path

import torch


class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, value):
        value = torch.nn.functional.max_pool2d(
            torch.nn.functional.relu(self.conv1(value)), 2
        )
        value = torch.nn.functional.max_pool2d(
            torch.nn.functional.relu(self.conv2(value)), 2
        )
        value = torch.flatten(value, 1)
        value = torch.nn.functional.relu(self.fc1(value))
        value = torch.nn.functional.relu(self.fc2(value))
        return self.fc3(value)


def initialize_deterministically(model):
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=parameter.dtype)
            values = ((values + index * 7).remainder(23) - 11) * 0.005
            parameter.copy_(values.reshape_as(parameter))


def main():
    model = LeNet5().eval()
    initialize_deterministically(model)
    example_input = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    exported = torch.export.export(model, (example_input,))
    torch.export.save(exported, Path(__file__).with_name("lenet5_pytorch.pt2"))


if __name__ == "__main__":
    main()
