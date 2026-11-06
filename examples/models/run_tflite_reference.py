"""Manual reference check for the MCUNet TFLite golden output.

This script is not part of the automated test suite. The E2E stress test
(`tests/std/tests/mcunet_tflite_owned.rs`) expects `[4, -5]` for an all-7 int8
input. That value matches the TFLite *reference kernels*; the XNNPACK delegate
rounds some fixed-point ops differently and prints `[5, -5]` on the same input,
so this script pins `BUILTIN_REF` to keep the reference reproducible.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL = Path(__file__).with_name("mcunet-10fps_vww.tflite")


def main() -> None:
    interpreter = tf.lite.Interpreter(
        model_path=str(MODEL),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_data = np.full(input_details["shape"], 7, dtype=np.int8)
    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()
    print(interpreter.get_tensor(output_details["index"]))


if __name__ == "__main__":
    main()
