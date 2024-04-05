# Copyright (c) 2024 Jonathan S. Pollack (https://github.com/JPPhoto)

from typing import Literal

import numpy as np
from PIL import Image

from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithMetadata,
    invocation,
)


@invocation("smart_seam", title="Smart Seam", tags=["image"], version="1.0.1")
class SmartSeamInvocation(BaseInvocation, WithMetadata):
    """Determines a smart seam between two images"""

    left_top_image: ImageField = InputField(description="The left or top image", title="Left/Top Image")
    right_bottom_image: ImageField = InputField(description="The right or bottom image", title="Right/Bottom Image")
    mode: Literal[("Left/Right", "Top/Bottom")] = InputField(default="Left/Right", description="Seam direction")

    def shift(self, arr, num, fill_value=255.0):
        result = np.empty_like(arr)
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr
        return result

    def get_seam_line(self, i1: Image, i2: Image, rotate: bool) -> Image:
        ia1 = np.array(i1) / 255.0
        if i1.mode != "L":
            ia1 = np.sum(ia1, -1) / 3.0

        ia2 = np.array(i2) / 255.0
        if i2.mode != "L":
            ia2 = np.sum(ia2, -1) / 3.0

        ia = ia2 - ia1

        if rotate:
            ia = np.rot90(ia, 1)

        # array is y by x
        max_y, max_x = ia.shape

        energy = abs(np.gradient(ia, axis=0)) + abs(np.gradient(ia, axis=1))

        res = np.copy(energy)

        for y in range(1, max_y):
            row = res[y, :]
            rowl = self.shift(row, -1)
            rowr = self.shift(row, 1)
            res[y, :] = res[y - 1, :] + np.min([row, rowl, rowr], axis=0)

        lowest_pos = int(max_x // 2)
        lowest_value = res[max_y - 1, lowest_pos]
        for x in range(0, max_x):
            candidate = res[max_y - 1, x]
            if candidate < lowest_value:
                lowest_pos = x
                lowest_value = candidate

        # create an array max_y long
        lowest_energy_line = np.empty([max_y], dtype="uint16")
        lowest_energy_line[max_y - 1] = lowest_pos

        for ypos in range(max_y - 2, -1, -1):
            lowest_pos = lowest_energy_line[ypos + 1]
            lpos = lowest_pos - 1 if lowest_pos > 1 else 0
            rpos = lowest_pos + 1 if lowest_pos < (max_x - 1) else (max_x - 1)
            yval = energy[ypos, lowest_pos]
            yvall = energy[ypos, lowest_pos - 1] if lowest_pos > 1 else np.Inf
            yvalr = energy[ypos, lowest_pos + 1] if lowest_pos < (max_x - 1) else np.Inf
            # if the value to the left is lower energy, pick that path...
            if yvall < yval and yvalr >= yval:
                lowest_pos = lpos
            # if the value to the right is lower energy, pick that path...
            if yvalr < yval and yvall >= yval:
                lowest_pos = rpos
            lowest_energy_line[ypos] = lowest_pos

        mask = np.zeros_like(ia)

        for ypos in range(0, max_y):
            to_fill = lowest_energy_line[ypos]
            mask[ypos, 0:to_fill] = 1

        if rotate:
            mask = np.rot90(mask, 3)

        image = Image.fromarray((mask * 255.0).astype("uint8"))

        return image

    def invoke(self, context: InvocationContext) -> ImageOutput:
        left_top_image = context.images.get_pil(self.left_top_image.image_name)
        left_top_image = left_top_image.convert("RGB")
        right_bottom_image = context.images.get_pil(self.right_bottom_image.image_name)
        right_bottom_image = right_bottom_image.convert("RGB")

        image = self.get_seam_line(left_top_image, right_bottom_image, self.mode == "Top/Bottom")

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)
