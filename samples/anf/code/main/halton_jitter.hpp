// Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause


#pragma once

#include <cstdint>
#include "glm/glm.hpp"

namespace Anf
{
    ////////////////////////////////////////////////////////////////////////////////
    // Class name: HaltonJitter
    ////////////////////////////////////////////////////////////////////////////////
    class HaltonJitter
    {
    public:
        /*
        * Computes a Halton sequence value for a given index and base.
        * The returned value is in [0, 1).
        * @param index : Sequence index (typically starts at 1).
        * @param base : Halton base (commonly 2 or 3).
        * @return Halton value in [0, 1).
        */
        static inline float Halton(uint32_t index, uint32_t base)
        {
            float    f = 1.0f;
            float    result = 0.0f;
            uint32_t i = index;

            while (i > 0)
            {
                f /= static_cast<float>(base);
                result += f * static_cast<float>(i % base);
                i /= base;
            }

            return result;
        }

        /*
        * Generates a 2D Halton jitter sample using bases 2 and 3.
        * Returned jitter is in pixel units in the range [-0.5, 0.5].
        * @param sample_index : Index of the sample (0..N-1).
        * @param sample_count : Number of samples in the repeating sequence (e.g., 8 or 16).
        * @return Jitter offset in pixel units, each component in [-0.5, 0.5].
        * @note We use (sample_index % sample_count) and add 1 because Halton is typically defined for index >= 1.
        */
        static inline glm::vec2 GetJitter(uint32_t sample_index, uint32_t sample_count)
        {
            const uint32_t wrapped = (sample_count > 0) ? (sample_index % sample_count) : 0;
            const uint32_t halton_i = wrapped + 1;

            const float jitter_x = Halton(halton_i, 2) - 0.5f;
            const float jitter_y = Halton(halton_i, 3) - 0.5f;

            return glm::vec2(jitter_x, jitter_y);
        }
    };

} // namespace Anf
